"""
Многозадачная нейросетевая модель (Multi-Task Learning) для классификации:
- Вид сорняка (Species Head): Бодяк полевой, Вьюнок полевой, Пырей ползучий
- Фаза вегетации (Stage Head): Розетка, Стеблевание
- Контроль неопределенности: перевод в статус unknown и review_required при низкой уверенности
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

SPECIES_NAMES = ["field_thistle", "field_bindweed", "couch_grass", "crop_wheat"]
SPECIES_RU = ["Бодяк полевой", "Вьюнок полевой", "Пырей ползучий", "Пшеница (Культура / Фон)"]
STAGE_NAMES = ["rosette", "stem_elongation"]
STAGE_RU = ["Розетка", "Стеблевание"]
DEFAULT_SPECIES_CONFIDENCE = 0.65
DEFAULT_STAGE_CONFIDENCE = 0.50


def infer_num_species_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    """Определяет число классов по последнему слою species head checkpoint-а."""
    for key, value in state_dict.items():
        if key.endswith("species_head.4.weight"):
            return int(value.shape[0])
    raise ValueError("В checkpoint не найден слой species_head.4.weight")


class FocalLoss(nn.Module):
    """
    Многоклассовый Focal Loss:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Фокусирует обучение на сложных примерах и устраняет дисбаланс/просадку recall по сложным классам (Вьюнок).
    """
    def __init__(
        self,
        alpha: Any = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        target_log_p = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_p = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - target_p) ** self.gamma
        loss = -focal_term * target_log_p

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(inputs.device)
            alpha_t = alpha_tensor.gather(0, targets)
            loss = loss * alpha_t

        if self.label_smoothing > 0:
            smooth_loss = -log_p.mean(dim=-1)
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-6)
            elif self.reduction == "sum":
                return loss.sum()
            return loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeedMultiTaskModel(nn.Module):
    """
    Архитектура с общим backbone (EfficientNet-B0 или MobileNetV3) и двумя независимыми головами:
    1. Species Head -> 4 класса (Бодяк, Вьюнок, Пырей, Пшеница/Культура)
    2. Growth Stage Head -> 2 класса (Розетка, Стеблевание)
    """
    def __init__(
        self,
        num_species: int = 4,
        num_stages: int = 2,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_species = num_species
        self.num_stages = num_stages

        if backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.features = base.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 1280
        elif backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_large(weights=weights)
            self.features = base.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 960
        else:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            self.features = base.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 576

        # Голова классификации вида сорняка/культуры
        self.species_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_species)
        )

        # Голова классификации фазы роста (для сорняков)
        self.stage_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_stages)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Извлечение глубоких эмбеддингов признаков (1280-d) до классификационных голов."""
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        species_logits = self.species_head(feat)
        stage_logits = self.stage_head(feat)
        return species_logits, stage_logits

    @torch.no_grad()
    def predict_crop(
        self,
        img_tensor: torch.Tensor,
        species_thresh: float = DEFAULT_SPECIES_CONFIDENCE,
        stage_thresh: float = DEFAULT_STAGE_CONFIDENCE
    ) -> Dict[str, Any]:
        self.eval()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        species_logits, stage_logits = self(img_tensor)

        species_probs = F.softmax(species_logits, dim=-1)[0]
        stage_probs = F.softmax(stage_logits, dim=-1)[0]

        sp_idx = int(torch.argmax(species_probs).item())
        sp_conf = float(species_probs[sp_idx].item())

        st_idx = int(torch.argmax(stage_probs).item())
        st_conf = float(stage_probs[st_idx].item())

        # Поддерживаем и 3-классовые, и 4-классовые checkpoint: подписи должны
        # соответствовать фактическому размеру выхода загруженной модели.
        active_species_names = SPECIES_NAMES[: species_probs.shape[0]]
        active_species_ru = SPECIES_RU[: species_probs.shape[0]]
        if not active_species_names:
            raise RuntimeError("Модель не вернула вероятности классов")

        sp_name = active_species_names[sp_idx]
        sp_ru = active_species_ru[sp_idx]
        st_name = STAGE_NAMES[st_idx]
        st_ru = STAGE_RU[st_idx]

        review_required = sp_conf < species_thresh

        # Единый confidence gate: класс показывается только при достижении
        # минимальной уверенности. Исходные вероятности остаются в ответе для
        # объяснимости, но сомнительный объект не получает автоматическое решение.
        if review_required:
            sp_name = "unknown"
            sp_ru = "Не определено"
            st_name = "unknown"
            st_ru = "Не определено"
            spray_action = "manual_review"

        # Защита от галлюцинаций: уверенно распознанная пшеница/культура
        elif sp_name == "crop_wheat":
            st_name = "not_applicable"
            st_ru = "Культура (Не применимо)"
            spray_action = "do_not_spray"
        else:
            spray_action = "spray_weed"
            # 1. Ограничение для пырея: фаза «розетка» невозможна в текущих данных
            if sp_name == "couch_grass":
                if st_idx == 0:
                    st_name = "unknown"
                    st_ru = "Не определено (нет эталона розетки)"
                    review_required = True
                else:
                    st_name = "stem_elongation"
                    st_ru = "Стеблевание"

            # 2. Порог уверенности фазы
            if st_conf < stage_thresh:
                st_name = "unknown"
                st_ru = "Не определено"
                review_required = True

            # Сомнительный объект нельзя автоматически отправлять в карту
            # опрыскивания: решение должен подтвердить агроном.
            if review_required:
                spray_action = "manual_review"

        return {
            "species": sp_name,
            "species_ru": sp_ru,
            "species_conf": round(sp_conf, 4),
            "stage": st_name,
            "stage_ru": st_ru,
            "stage_conf": round(st_conf, 4),
            "spray_action": spray_action,
            "review_required": review_required,
            "all_species_probs": {
                name: round(float(prob), 4)
                for name, prob in zip(active_species_names, species_probs)
            },
            "all_stage_probs": {STAGE_NAMES[i]: round(float(stage_probs[i]), 4) for i in range(len(STAGE_NAMES))}
        }
