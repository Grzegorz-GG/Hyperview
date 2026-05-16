import torch
import torch.nn as nn
import timm


class HyperspectralRegressor(nn.Module):
    """Hyperspectral regression model.

    PCA reduces the 150 spectral bands to ``in_channels`` components upstream;
    this model receives those PCA components directly — no 1×1 conv reducer.

    Args:
        in_channels   : number of PCA components (= data.pca_components)
        n_outputs     : number of regression targets (4: P, K, Mg, pH)
        backbone_name : timm model name
        pretrained    : load ImageNet / in22k weights (only valid when in_channels == 3)
        dropout       : dropout rate in regression head
    """

    def __init__(
        self,
        in_channels: int   = 3,
        n_outputs: int     = 4,
        backbone_name: str = "convnext_small_in22k",
        pretrained: bool   = True,
        dropout: float     = 0.3,
    ):
        super().__init__()

        # Pretrained weights only make sense for 3-channel input (ImageNet/in22k)
        effective_pretrained = pretrained and (in_channels == 3)
        if pretrained and in_channels != 3:
            print(
                f"[HyperspectralRegressor] pretrained=True ignored for "
                f"in_channels={in_channels} (not 3). Training from scratch."
            )

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=effective_pretrained,
            in_chans=in_channels,
            num_classes=0,
        )

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            raise RuntimeError(
                f"Cannot determine feature dimension for backbone '{backbone_name}'. "
                "Ensure the model exposes num_features."
            )

        # +1 for the log(valid_pixel_count) patch-size feature
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim + 1, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x: torch.Tensor, num_valid_pixels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x                : (B, in_channels, H, W) — invalid pixels are zero
            num_valid_pixels : (B,) count of valid (non-padded) pixels per patch
        Returns:
            (B, n_outputs)
        """
        features  = self.backbone(x)                                        # (B, feat_dim)
        size_feat = torch.log1p(num_valid_pixels.float()) / 10.0            # (B,)  ~[0,1]
        features  = torch.cat([features, size_feat.unsqueeze(1)], dim=1)   # (B, feat_dim+1)
        return self.regressor(features)
