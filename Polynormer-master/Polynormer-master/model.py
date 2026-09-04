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


class Graph2MapClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_chans: int = 4,
        backbone_name: str = "resnet18",
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
            backbone_name: Vision backbone from timm (e.g. 'resnet18', 'resnet34', 'convnext_tiny', 'efficientnet_b0').
            pretrained: Whether to load ImageNet pre-trained weights (adapted for in_chans=4).
            node_feat_dim: Dimension of raw node feature vector (if 0 or use_node_features=False, pure vision is used).
            use_node_features: Whether to fuse raw target node feature vector with the visual topology embedding.
            hidden_dim: Hidden dimension for fusion and classifier head.
            dropout: Dropout probability.
            drop_rate: Backbone internal drop rate if supported.
        """
        super().__init__()
        if timm is None:
            raise ImportError(
                "The 'timm' package is required for Graph2MapClassifier. "
                "Install it using: pip install timm"
            )

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.node_feat_dim = node_feat_dim
        self.use_node_features = use_node_features and (node_feat_dim > 0)
        self.hidden_dim = hidden_dim

        # 1. Instantiate Vision Backbone using timm
        # num_classes=0 returns the pooled feature embedding vector
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            drop_rate=drop_rate
        )

        # Retrieve visual embedding dimension
        self.vis_dim = self.backbone.num_features

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
        vis_emb = self.backbone(ego_maps)  # (B, vis_dim)

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
    node_feat_dim: int = 0,
    backbone_name: str = "resnet18",
    pretrained: bool = False,
    use_node_features: bool = True,
    hidden_dim: int = 256,
    dropout: float = 0.3
) -> Graph2MapClassifier:
    """
    Helper function to instantiate Graph2MapClassifier.
    """
    return Graph2MapClassifier(
        num_classes=num_classes,
        in_chans=4,
        backbone_name=backbone_name,
        pretrained=pretrained,
        node_feat_dim=node_feat_dim,
        use_node_features=use_node_features,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
