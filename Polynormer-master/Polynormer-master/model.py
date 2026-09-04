"""
Graph Vision Architectures for Graph-as-an-Image Learning.
Pure PyTorch implementation (Zero external timm dependencies for instant import).

Models:
1. GraphSpectrogramNet: Time-Frequency 2D ConvNet tailored for (C, num_hops, num_bands)
   Graph Spectrograms with SpecAugment (hop & band masking).
2. GraphEgoMapNet: Spatial 2D CNN tailored for (C, resolution, resolution)
   Ego-Centric continuous topological maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. SPECAUGMENT FOR GRAPH SPECTROGRAMS
# =============================================================================

class SpecAugment(nn.Module):
    """SpecAugment for Graph Spectrograms: random hop masking & frequency band masking."""
    def __init__(self, hop_mask_max: int = 2, freq_mask_max: int = 16):
        super().__init__()
        self.hop_mask_max = hop_mask_max
        self.freq_mask_max = freq_mask_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        b, c, h, w = x.shape
        # Hop horizon masking
        if self.hop_mask_max > 0 and h > self.hop_mask_max:
            h_len = torch.randint(1, self.hop_mask_max + 1, (1,)).item()
            h_start = torch.randint(0, max(1, h - h_len), (1,)).item()
            x = x.clone()
            x[:, :, h_start:h_start + h_len, :] = 0.0
        # Frequency band masking
        if self.freq_mask_max > 0 and w > self.freq_mask_max:
            f_len = torch.randint(1, self.freq_mask_max + 1, (1,)).item()
            f_start = torch.randint(0, max(1, w - f_len), (1,)).item()
            x = x.clone()
            x[:, :, :, f_start:f_start + f_len] = 0.0
        return x


# =============================================================================
# 2. GRAPH SPECTROGRAM CONVNET (Time-Frequency Vision Backbone)
# =============================================================================

class GraphSpectrogramNet(nn.Module):
    """
    Time-Frequency 2D ConvNet specifically optimized for Graph Spectrogram inputs.
    Uses rectangular kernels (capturing multi-hop temporal evolution across frequency bands)
    with residual pooling and adaptive global spectral aggregation.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        node_feat_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_node_features: bool = True
    ):
        super().__init__()
        self.use_node_features = use_node_features and (node_feat_dim > 0)
        self.spec_augment = SpecAugment(hop_mask_max=2, freq_mask_max=16)

        # Time-Frequency Spectrogram Vision Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(dropout)

        # Raw node feature projection (optional fusion)
        if self.use_node_features:
            self.feat_mlp = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            classifier_in = hidden_dim * 2
        else:
            self.feat_mlp = None
            classifier_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spec: torch.Tensor, raw_x: torch.Tensor = None) -> torch.Tensor:
        x = self.spec_augment(spec)
        v_emb = self.dropout(self.backbone(x))

        if self.use_node_features and raw_x is not None:
            f_emb = self.feat_mlp(raw_x)
            fused = torch.cat([v_emb, f_emb], dim=1)
        else:
            fused = v_emb

        return self.classifier(fused)


# =============================================================================
# 3. GRAPH EGO-MAP CONVNET (Spatial Cartography Vision Backbone)
# =============================================================================

class GraphEgoMapNet(nn.Module):
    """
    Compact, specialized 2D CNN backbone tailored for 128x128 multi-channel continuous Ego-Maps.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 8,
        node_feat_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_node_features: bool = True
    ):
        super().__init__()
        self.use_node_features = use_node_features and (node_feat_dim > 0)

        # 4-stage Spatial CNN Backbone
        self.backbone = nn.Sequential(
            # Stage 1: 128x128 -> 64x64
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Stage 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Stage 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Stage 4: 16x16 -> Global Average Pooling
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(dropout)

        if self.use_node_features:
            self.feat_mlp = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            classifier_in = hidden_dim * 2
        else:
            self.feat_mlp = None
            classifier_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ego_map: torch.Tensor, raw_x: torch.Tensor = None) -> torch.Tensor:
        v_emb = self.dropout(self.backbone(ego_map))

        if self.use_node_features and raw_x is not None:
            f_emb = self.feat_mlp(raw_x)
            fused = torch.cat([v_emb, f_emb], dim=1)
        else:
            fused = v_emb

        return self.classifier(fused)


# Backward-compatibility aliases
SpectrogramCNN = GraphSpectrogramNet
LightEgoNet = GraphEgoMapNet
Graph2MapClassifier = GraphEgoMapNet


# =============================================================================
# 4. UNIFIED MODEL FACTORY
# =============================================================================

def create_vision_model(
    num_classes: int,
    in_chans: int = 3,
    node_feat_dim: int = 0,
    backbone_name: str = "spectrogram_cnn",
    pretrained: bool = False,
    use_node_features: bool = True,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    representation: str = "spectrogram"
) -> nn.Module:
    """
    Unified factory function to instantiate GraphSpectrogramNet or GraphEgoMapNet.
    """
    if representation == "spectrogram" or backbone_name in ("spectrogram_cnn", "spectrogram_net", "spectrogram"):
        return GraphSpectrogramNet(
            in_channels=in_chans,
            num_classes=num_classes,
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_node_features=use_node_features
        )
    else:
        return GraphEgoMapNet(
            in_channels=in_chans,
            num_classes=num_classes,
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_node_features=use_node_features
        )
