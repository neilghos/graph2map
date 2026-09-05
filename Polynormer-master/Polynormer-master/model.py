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

class GraphSpectrogramNet(nn.Module):
    """
    Time-Frequency 2D ConvNet specifically optimized for Graph Spectrogram inputs.
    Uses rectangular kernels (capturing multi-hop temporal evolution across frequency bands)
    with residual pooling and adaptive global spectral aggregation.
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 8,
        node_feat_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_node_features: bool = True
    ):
        super().__init__()
        self.use_node_features = use_node_features and (node_feat_dim > 0)

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
        v_emb = self.dropout(self.backbone(spec))

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
    in_chans: int = 6,
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
    if representation in ("atlas", "atlas_7ch", "atlas_10ch", "ego_map") or backbone_name in ("egocnn", "graph_ego_map_net", "ego"):
        return GraphEgoMapNet(
            in_channels=in_chans,
            num_classes=num_classes,
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_node_features=use_node_features
        )
    else:
        return GraphSpectrogramNet(
            in_channels=in_chans,
            num_classes=num_classes,
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_node_features=use_node_features
        )
