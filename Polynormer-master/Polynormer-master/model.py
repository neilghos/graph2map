"""
Graph2Map Vision Classifier Models (via timm).
Translates 4-channel node-level Ego-Maps into node class predictions.

Supports:
- Pure Visual Mode: Uses 2D vision backbones (ResNet, ConvNeXt, EfficientNet, MobileNet, ViT)
  directly on the 4-channel continuous topological Ego-Maps (Ego-Density, Subgraph Edges, PPR Diffusion, Feature Homophily).
- Hybrid Multimodal Mode: Fuses the visual topological embedding with the target node's raw feature vector
  via an MLP fusion head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


class LightEgoNet(nn.Module):
    """
    Compact, specialized 2D CNN backbone tailored for 128x128 multi-channel continuous + canonical Ego-Maps.
    Keeps parameter counts in the ~100k - ~580k range (matching standard Graph Neural Network
    parameter budgets) to eliminate overfitting on graph benchmark datasets.
    """
    def __init__(
        self,
        in_chans: int = 4,
        out_dim: int = 256,
        dropout: float = 0.3,
        width_multiplier: float = 1.0
    ):
        super().__init__()
        c1 = max(16, int(32 * width_multiplier))
        c2 = max(32, int(64 * width_multiplier))
        c3 = max(64, int(128 * width_multiplier))
        c4 = max(128, int(256 * width_multiplier))

        self.num_features = c4

        # Stage 1: 128x128 -> 64x64
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_chans, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.3)
        )

        # Stage 2: 64x64 -> 32x32
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.4)
        )

        # Stage 3: 32x32 -> 16x16
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5)
        )

        # Stage 4: 16x16 -> 8x8 -> Global Average Pooling
        self.stage4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class Graph2MapClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_chans: int = 4,
        backbone_name: str = "egocnn",
        pretrained: bool = False,
        node_feat_dim: int = 0,
        use_node_features: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        drop_rate: float = 0.0
    ):
        """
        Args:
            num_classes: Number of target node classes.
            in_chans: Number of input Ego-Map channels (default 4).
            backbone_name: Vision backbone name. Supports:
                - 'egocnn' / 'lightegonet': Custom tight 4-stage CNN (~580K params).
                - 'tiny_egocnn': Ultra-tight 4-stage CNN (~145K params, matching classic GNN scale).
                - Any timm backbone (e.g. 'mobilenetv3_small_050', 'mobilenetv3_small_100', 'resnet18').
            pretrained: Whether to load ImageNet pre-trained weights (adapted for in_chans=4).
            node_feat_dim: Dimension of raw node feature vector (if 0 or use_node_features=False, pure vision is used).
            use_node_features: Whether to fuse raw target node feature vector with the visual topology embedding.
            hidden_dim: Hidden dimension for fusion and classifier head.
            dropout: Dropout probability.
            drop_rate: Backbone internal drop rate if supported.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.node_feat_dim = node_feat_dim
        self.use_node_features = use_node_features and (node_feat_dim > 0)
        self.hidden_dim = hidden_dim

        # 1. Instantiate Vision Backbone
        if backbone_name in ('egocnn', 'lightegonet', 'light_cnn'):
            self.backbone = LightEgoNet(in_chans=in_chans, out_dim=hidden_dim, dropout=dropout, width_multiplier=1.0)
            self.vis_dim = self.backbone.num_features
        elif backbone_name in ('tiny_egocnn', 'tiny_cnn', 'micro_egocnn'):
            self.backbone = LightEgoNet(in_chans=in_chans, out_dim=hidden_dim, dropout=dropout, width_multiplier=0.5)
            self.vis_dim = self.backbone.num_features
        else:
            if timm is None:
                raise ImportError(
                    "The 'timm' package is required for timm-based backbones. "
                    "Install it using: pip install timm"
                )
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                drop_rate=dropout if drop_rate == 0.0 else drop_rate
            )
            self.vis_dim = self.backbone.num_features

        self.vis_dropout = nn.Dropout(dropout)

        # 2. Target Node Feature Encoder (if hybrid mode is active)
        if self.use_node_features:
            self.node_encoder = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            fusion_in_dim = self.vis_dim + hidden_dim
        else:
            self.node_encoder = None
            fusion_in_dim = self.vis_dim

        # 3. Final Classification Head
        self.head = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def reset_parameters(self):
        """
        Reset model parameters to support standard multi-run benchmark loops.
        """
        # Re-initialize non-backbone modules
        if self.node_encoder is not None:
            for m in self.node_encoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Re-initialize backbone weights
        for m in self.backbone.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        ego_maps: torch.Tensor,
        node_feats: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            ego_maps: 4D Tensor of shape (B, 4, H, W) containing continuous 2D Ego-Maps.
            node_feats: Optional 2D Tensor of shape (B, node_feat_dim) containing target node raw attributes.
        Returns:
            Logits Tensor of shape (B, num_classes).
        """
        # 1. Extract visual embedding from 4-channel Ego-Map
        vis_emb = self.vis_dropout(self.backbone(ego_maps))  # (B, vis_dim)

        # 2. Fuse with raw target node attributes if enabled
        if self.use_node_features and node_feats is not None:
            node_emb = self.node_encoder(node_feats)  # (B, hidden_dim)
            combined = torch.cat([vis_emb, node_emb], dim=-1)  # (B, vis_dim + hidden_dim)
        else:
            combined = vis_emb

        # 3. Predict class logits
        logits = self.head(combined)  # (B, num_classes)
        return logits


def create_vision_model(
    num_classes: int,
    in_chans: int = 4,
    node_feat_dim: int = 0,
    backbone_name: str = "egocnn",
    pretrained: bool = False,
    use_node_features: bool = True,
    hidden_dim: int = 256,
    dropout: float = 0.5
) -> Graph2MapClassifier:
    """
    Helper function to instantiate Graph2MapClassifier.
    """
    return Graph2MapClassifier(
        num_classes=num_classes,
        in_chans=in_chans,
        backbone_name=backbone_name,
        pretrained=pretrained,
        node_feat_dim=node_feat_dim,
        use_node_features=use_node_features,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
