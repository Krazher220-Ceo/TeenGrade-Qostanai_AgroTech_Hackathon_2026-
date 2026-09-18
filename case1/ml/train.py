"""
Скрипт глубокого обучения многозадачной модели классификации сорняков (вид + фаза).
"""

import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report

from case1.ml.dataset import WeedMultiTaskDataset, SPECIES_TO_IDX, STAGE_TO_IDX
from case1.ml.multitask_model import WeedMultiTaskModel, FocalLoss, SPECIES_NAMES, STAGE_NAMES


def train_model(
    manifest_path: str = "case1/data/manifest_v2.csv",
    models_dir: str = "case1/models",
    output_dir: str = "case1/output",
    backbone_name: str = "efficientnet_b0",
    epochs: int = 35,
    batch_size: int = 16,
    lr: float = 8e-4,
    use_focal: bool = True,
    device_str: str = "auto"
):
    # Если есть /data/Сорняки, используем или строим полный 10k манифест
    server_weeds = Path("/data/Сорняки")
    full_manifest = Path("case1/data/manifest_full_10k.csv")
    if server_weeds.exists() and not full_manifest.exists():
        from case1.ml.build_full_dataset import build_manifest
        manifest_path = str(build_manifest(output_path=full_manifest))
    elif full_manifest.exists():
        manifest_path = str(full_manifest)
    elif not Path(manifest_path).exists() and Path("case1/data/manifest.csv").exists():
        manifest_path = "case1/data/manifest.csv"

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    # Оптимизация под NVIDIA H100 (20GB VRAM): батч 64 ускоряет обучение в 4 раза
    if device.type == "cuda" and batch_size <= 16:
        batch_size = 64
        num_workers = 2
    else:
        num_workers = 0

    print(f"=== ОБУЧЕНИЕ МНОГОЗАДАЧНОЙ МОДЕЛИ ({backbone_name.upper()}, {epochs} ЭПОХ, Device: {device}, Batch: {batch_size}) ===")
    print(f"=== В соответствии со шпаргалкой агронома «Олжа Агро» (Классы A/B, 3 фазы, ЭПВ) ===")

    # Расширенная агро-аугментация
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random")
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = WeedMultiTaskDataset(manifest_path, split="train", transform=train_transform)
    val_ds = WeedMultiTaskDataset(manifest_path, split="val", transform=eval_transform, species_to_idx=train_ds.species_to_idx)
    test_ds = WeedMultiTaskDataset(manifest_path, split="test", transform=eval_transform, species_to_idx=train_ds.species_to_idx)

    print(f"Выборки ({manifest_path}): train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Классов сорняков: {len(train_ds.species_to_idx)} | Фаз вегетации: 3 (по шпаргалке)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    num_classes = len(train_ds.species_to_idx)
    num_stages = 3  # 3 окна по шпаргалке: 0=семядоли-2листа, 1=4-6листьев, 2=цветение/пропущено

    # Сохраняем маппинг классов для инференса
    mapping_data = {
        "species_to_idx": train_ds.species_to_idx,
        "idx_to_species": train_ds.idx_to_species,
        "num_classes": num_classes,
        "num_stages": num_stages,
        "stage_names": ["cotyledon_to_2_leaves", "4_to_6_leaves", "over_6_leaves_or_flowering"],
        "stage_ru": ["Семядоли — 2 листа", "4–6 листьев", "Более 6 листьев / цветение"],
    }
    with open(models_path / "species_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)

    model = WeedMultiTaskModel(
        num_species=num_classes,
        num_stages=num_stages,
        backbone_name=backbone_name,
        pretrained=True,
        dropout=0.3
    ).to(device)

    species_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    stage_criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_score = -1.0
    best_weights_path = models_path / "multitask_weeds_best.pt"
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        sp_correct, st_correct, st_total = 0, 0, 0

        for imgs, sp_targets, st_targets, st_masks, _ in train_loader:
            imgs = imgs.to(device)
            sp_targets = sp_targets.to(device)
            st_targets = st_targets.to(device)
            st_masks = st_masks.to(device)

            optimizer.zero_grad()
            sp_logits, st_logits = model(imgs)

            l_sp = species_criterion(sp_logits, sp_targets)

            l_st_raw = stage_criterion(st_logits, st_targets)
            if st_masks.sum() > 0:
                l_st = (l_st_raw * st_masks).sum() / (st_masks.sum() + 1e-6)
            else:
                l_st = torch.tensor(0.0, device=device)

            loss = l_sp + 0.85 * l_st
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(imgs)

            sp_preds = torch.argmax(sp_logits, dim=-1)
            sp_correct += (sp_preds == sp_targets).sum().item()

            st_preds = torch.argmax(st_logits, dim=-1)
            valid_mask = (st_masks > 0)
            if valid_mask.sum() > 0:
                st_correct += (st_preds[valid_mask] == st_targets[valid_mask]).sum().item()
                st_total += valid_mask.sum().item()

        scheduler.step()
        train_loss /= len(train_ds)
        train_sp_acc = sp_correct / len(train_ds)
        train_st_acc = st_correct / (st_total + 1e-6)

        val_metrics = evaluate_loader(model, val_loader, device)
        val_sp_f1 = val_metrics["species_macro_f1"]
        val_st_f1 = val_metrics["stage_macro_f1"]
        combined_val = 0.6 * val_sp_f1 + 0.4 * val_st_f1

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_species_acc": round(train_sp_acc, 4),
            "val_species_f1": round(val_sp_f1, 4),
            "val_stage_f1": round(val_st_f1, 4)
        })

        if combined_val > best_val_score:
            best_val_score = combined_val
            torch.save(model.state_dict(), best_weights_path)
            mark = "★ BEST"
        else:
            mark = ""

        if epoch % 5 == 0 or epoch == 1 or mark:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Val Sp F1: {val_sp_f1:.3f} | Val St F1: {val_st_f1:.3f} {mark}")

    total_time = time.time() - start_time
    print(f"\nОбучение завершено за {total_time:.1f} с. Лучшая модель сохранена в {best_weights_path}")

    # Финальная оценка на TEST наборе
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_metrics = evaluate_loader(model, test_loader, device, detailed=True)

    print("\n================================================================================")
    print(f"ИТОГОВЫЕ МЕТРИКИ НА ОТЛОЖЕННОМ ТЕСТОВОМ НАБОРЕ (TEST SPLIT - {len(test_ds)} ФОТО):")
    print("================================================================================")
    print(f"Вид сорняка (Species):")
    print(f"  Accuracy:       {test_metrics['species_accuracy'] * 100:.1f}%")
    print(f"  Macro-F1:       {test_metrics['species_macro_f1']:.3f}")
    print(f"  Recall по классам:")
    for k, v in test_metrics["species_recall_per_class"].items():
        print(f"    - {k}: {v * 100:.1f}%")

    print(f"\nФаза роста (Stage):")
    print(f"  Accuracy:       {test_metrics['stage_accuracy'] * 100:.1f}%")
    print(f"  Macro-F1:       {test_metrics['stage_macro_f1']:.3f}")

    # Копирование в multitask_weeds_focal.pt
    if use_focal:
        torch.save(model.state_dict(), models_path / "multitask_weeds_focal.pt")

    results_report = {
        "backbone": backbone_name,
        "loss_type": "focal" if use_focal else "crossentropy",
        "training_time_s": round(total_time, 2),
        "epochs": epochs,
        "best_val_score": round(best_val_score, 4),
        "dataset_sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "test_metrics": test_metrics,
        "history": history
    }
    with open(out_path / "classifier_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=2, ensure_ascii=False)

    return model, results_report


def evaluate_loader(model, loader, device, detailed: bool = False):
    model.eval()
    sp_true, sp_pred = [], []
    st_true, st_pred = [], []

    with torch.no_grad():
        for imgs, sp_targets, st_targets, st_masks, _ in loader:
            imgs = imgs.to(device)
            sp_logits, st_logits = model(imgs)

            sp_p = torch.argmax(sp_logits, dim=-1).cpu().numpy()
            sp_true.extend(sp_targets.numpy())
            sp_pred.extend(sp_p)

            st_p = torch.argmax(st_logits, dim=-1).cpu().numpy()
            valid_idx = (st_masks > 0).numpy()
            if valid_idx.sum() > 0:
                st_true.extend(st_targets.numpy()[valid_idx])
                st_pred.extend(st_p[valid_idx])

    sp_acc = accuracy_score(sp_true, sp_pred)
    sp_f1 = f1_score(sp_true, sp_pred, average="macro", zero_division=0)

    st_acc = accuracy_score(st_true, st_pred) if st_true else 0.0
    st_f1 = f1_score(st_true, st_pred, average="macro", zero_division=0) if st_true else 0.0

    metrics = {
        "species_accuracy": round(float(sp_acc), 4),
        "species_macro_f1": round(float(sp_f1), 4),
        "stage_accuracy": round(float(st_acc), 4),
        "stage_macro_f1": round(float(st_f1), 4)
    }

    if detailed:
        target_names = [loader.dataset.idx_to_species.get(i, f"class_{i}") for i in range(len(loader.dataset.idx_to_species))]
        rep = classification_report(sp_true, sp_pred, target_names=target_names, output_dict=True, zero_division=0)
        metrics["species_recall_per_class"] = {
            name: round(rep[name]["recall"], 4)
            for name in target_names if name in rep
        }

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train multi-task weed classifier")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Backbone name")
    parser.add_argument("--no-focal", action="store_true", help="Disable Focal Loss")
    args = parser.parse_args()

    train_model(
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_focal=not args.no_focal
    )
