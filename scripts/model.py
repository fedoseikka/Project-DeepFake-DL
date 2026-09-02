import torch
import torch.nn as nn
from torchvision import models

from config import CNN_FEATURE_DIM, LSTM_HIDDEN_DIM, TRANSFORMER_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT

class CNNFeatureExtractor(nn.Module):
    """Извлечение пространственных признаков с помощью EfficientNet-B0"""
    def __init__(self, feature_dim=CNN_FEATURE_DIM):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.projection = nn.Linear(1280, feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.projection(features)


class HybridRNNTransformerDeepfake(nn.Module):
    """
    Гибридная модель для классификации deepfake:
    - CNN (EfficientNet) для извлечения пространственных признаков
    - BiLSTM для локальных временных зависимостей
    - Transformer для глобальных временных связей
    """
    def __init__(self,
                 cnn_feature_dim=CNN_FEATURE_DIM,
                 lstm_hidden_dim=LSTM_HIDDEN_DIM,
                 transformer_dim=TRANSFORMER_DIM,
                 num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()

        # 1. CNN-экстрактор
        self.cnn_extractor = CNNFeatureExtractor(feature_dim=cnn_feature_dim)

        # 2. BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.lstm_projection = nn.Linear(lstm_hidden_dim * 2, transformer_dim)
        self.lstm_dropout = nn.Dropout(dropout)

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_dropout = nn.Dropout(dropout)

        # 4. Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Извлечение CNN-признаков
        cnn_features = self.cnn_extractor(x)          # [batch, cnn_feature_dim]
        seq_len = 8
        seq_features = cnn_features.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, cnn_feature_dim]
        noise = torch.randn_like(seq_features) * 0.01
        seq_features = seq_features + noise

        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(seq_features)
        lstm_pooled = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # [batch, 2*lstm_hidden]
        lstm_projected = self.lstm_projection(lstm_pooled)
        lstm_projected = self.lstm_dropout(lstm_projected)

        # Transformer (глобальные зависимости)
        trans_out = self.transformer(seq_features)   # [batch, seq_len, transformer_dim]
        trans_pooled = trans_out.mean(dim=1)         # [batch, transformer_dim]
        trans_pooled = self.transformer_dropout(trans_pooled)

        # Объединение и классификация
        combined = torch.cat((lstm_projected, trans_pooled), dim=1)
        logits = self.classifier(combined)
        return logits
