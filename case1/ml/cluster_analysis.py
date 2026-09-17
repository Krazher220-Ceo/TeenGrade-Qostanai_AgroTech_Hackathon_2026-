"""
Скрипт кластерного анализа эмбеддингов (t-SNE / PCA / Silhouette Score)
для доказательства изолированности кластера пшеницы (защита от галлюцинаций)
и сохранения различимости сорняков (бодяк, вьюнок, пырей).
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine, euclidean

from case1.ml.dataset import WeedMultiTaskDataset, SPECIES_TO_IDX
from case1.ml.multitask_model import WeedMultiTaskModel, SPECIES_NAMES, SPECIES_RU

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = ROOT_DIR / "case1" / "data" / "manifest_v2.csv"
MODEL_PATH = ROOT_DIR / "case1" / "models" / "multitask_weeds_best.pt"
OUTPUT_JSON = ROOT_DIR / "case1" / "output" / "clusters_2d.json"


def run_cluster_analysis(
    manifest_path: str = str(MANIFEST_PATH),
    model_path: str = str(MODEL_PATH),
    output_path: str = str(OUTPUT_JSON),
    device_str: str = "auto"
) -> Dict[str, Any]:
    print("=== КЛАСТЕРНЫЙ АНАЛИЗ ГЛУБОКИХ ЭМБЕДДИНГОВ (t-SNE / SILHOUETTE) ===")

    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Загружаем тестовую и валидационную выборку (или весь датасет для полной картины)
    val_ds = WeedMultiTaskDataset(manifest_path, split="val", transform=transform)
    test_ds = WeedMultiTaskDataset(manifest_path, split="test", transform=transform)

    # Объединяем контрольные сэмплы
    eval_samples = val_ds.samples + test_ds.samples
    print(f"Всего анализируемых контрольных образцов: {len(eval_samples)}")

    model = WeedMultiTaskModel(num_species=len(SPECIES_NAMES), num_stages=2, backbone_name="efficientnet_b0", pretrained=False)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Загружены веса из {model_path}")
    else:
        print(f"[WARN] Веса не найдены по пути {model_path}, используются дефолтные")

    model.to(device)
    model.eval()

    all_embeddings = []
    all_labels = []
    metadata = []

    print("Извлечение 1280-мерных векторов признаков...")
    with torch.no_grad():
        for s in eval_samples:
            try:
                from PIL import Image
                img = Image.open(s["path"]).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                feat = model.extract_features(tensor)[0]
                feat_norm = F.normalize(feat, p=2, dim=0).cpu().numpy()

                all_embeddings.append(feat_norm)
                all_labels.append(s["species_idx"])
                metadata.append(s)
            except Exception:
                continue

    X = np.array(all_embeddings)
    y = np.array(all_labels)
    print(f"Матрица эмбеддингов: {X.shape}")

    # Расчет метрики разделимости кластеров (Silhouette Coefficient)
    sil_score = float(silhouette_score(X, y))
    print(f"★ Silhouette Score (качество кластеризации): {sil_score:.4f} (норма: >0.35)")

    # Расчет центроидов для каждого из 4 классов
    centroids = {}
    for sp_name, sp_idx in SPECIES_TO_IDX.items():
        idx_mask = (y == sp_idx)
        if idx_mask.sum() > 0:
            centroids[sp_name] = X[idx_mask].mean(axis=0)

    # Расстояния между центроидом Пшеницы и сорняками
    centroid_separations = {}
    if "crop_wheat" in centroids:
        c_wheat = centroids["crop_wheat"]
        for sp_name in ["field_thistle", "field_bindweed", "couch_grass"]:
            if sp_name in centroids:
                c_other = centroids[sp_name]
                cos_dist = float(cosine(c_wheat, c_other))
                euc_dist = float(euclidean(c_wheat, c_other))
                centroid_separations[f"wheat_vs_{sp_name}"] = {
                    "cosine_distance": round(cos_dist, 4),
                    "euclidean_distance": round(euc_dist, 4),
                    "is_well_separated": cos_dist > 0.40
                }
                print(f"Разделение Пшеница vs {sp_name}: Cosine Dist = {cos_dist:.3f}")

    # Снижение размерности t-SNE для интерактивного 2D дашборда
    print("Вычисление 2D проекции t-SNE (perplexity=25)...")
    tsne = TSNE(n_components=2, perplexity=min(25, len(X) - 1), random_state=42, max_iter=1000)
    X_2d_tsne = tsne.fit_transform(X)

    # Снижение размерности PCA для линейного сравнения
    pca = PCA(n_components=2, random_state=42)
    X_2d_pca = pca.fit_transform(X)

    # Формирование JSON для веб-визуализации
    points = []
    for i in range(len(X)):
        sp_idx = int(y[i])
        sp_name = SPECIES_NAMES[sp_idx] if sp_idx < len(SPECIES_NAMES) else "unknown"
        sp_ru = SPECIES_RU[sp_idx] if sp_idx < len(SPECIES_RU) else "Неизвестно"
        points.append({
            "tsne_x": round(float(X_2d_tsne[i, 0]), 3),
            "tsne_y": round(float(X_2d_tsne[i, 1]), 3),
            "pca_x": round(float(X_2d_pca[i, 0]), 3),
            "pca_y": round(float(X_2d_pca[i, 1]), 3),
            "species": sp_name,
            "species_ru": sp_ru,
            "filename": metadata[i].get("filename", ""),
            "path": metadata[i].get("path", "")
        })

    cluster_report = {
        "num_samples": len(points),
        "embedding_dim": int(X.shape[1]),
        "silhouette_score": round(sil_score, 4),
        "centroid_separations": centroid_separations,
        "points": points
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cluster_report, f, indent=2, ensure_ascii=False)

    print(f"★ Кластерный отчет успешно сохранен в {out_file}")
    return cluster_report


if __name__ == "__main__":
    run_cluster_analysis()
