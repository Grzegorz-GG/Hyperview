import torch
import torch.nn as nn
import timm


class HyperspectralRegressor(nn.Module):
    """Hyperspectral regression model supporting two spectral reduction strategies.

    **PCA mode** (``use_spectral_reducer=False``, default):
        Spectral reduction is done offline by PCA before training.
        The model receives ``in_channels`` PCA components directly.
        Pretrained ImageNet/in22k weights are usable when ``in_channels == 3``.

    **Conv mode** (``use_spectral_reducer=True``):
        A learnable 1×1 conv stack reduces raw 150-band input to ``in_channels``
        inside the model:  raw_in_channels → reducer_mid_channels → in_channels.
        Pretrained ImageNet/in22k weights are usable when ``in_channels == 3``.

    Args:
        in_channels          : channels the backbone sees
                               (= pca_components in PCA mode,
                                = reducer_channels in conv mode)
        n_outputs            : number of regression targets (4: P, K, Mg, pH)
        backbone_name        : timm model name
        pretrained           : load ImageNet / in22k weights
                               (only effective when in_channels == 3)
        dropout              : dropout rate in regression head
        use_spectral_reducer : if True, prepend learnable conv spectral reducer
        raw_in_channels      : input spectral bands fed to the reducer (default 150)
        reducer_mid_channels : intermediate width in reducer: raw → mid → in_channels
    """

    def __init__(
        self,
        in_channels: int           = 3,
        n_outputs: int             = 4,
        backbone_name: str         = "convnext_small_in22k",
        pretrained: bool           = True,
        dropout: float             = 0.3,
        use_spectral_reducer: bool = False,
        raw_in_channels: int       = 150,
        reducer_mid_channels: int  = 32,
    ):
        super().__init__()

        self.use_spectral_reducer = use_spectral_reducer

        # ---- optional conv spectral reducer (conv mode only) ---------------
        if use_spectral_reducer:
            self.spectral_reducer = nn.Sequential(
                nn.Conv2d(raw_in_channels, reducer_mid_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(reducer_mid_channels),
                nn.GELU(),
                nn.Conv2d(reducer_mid_channels, in_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
            )

        # ---- backbone ------------------------------------------------------
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

        # ---- regression head -----------------------------------------------
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
            x                : (B, C_in, H, W) — invalid pixels are zero
                               C_in = raw_in_channels (conv mode) or in_channels (PCA mode)
            num_valid_pixels : (B,) count of valid (non-padded) pixels per patch
        Returns:
            (B, n_outputs)
        """
        if self.use_spectral_reducer:
            x = self.spectral_reducer(x)             # (B, in_channels, H, W)

        features  = self.backbone(x)                                        # (B, feat_dim)
        size_feat = torch.log1p(num_valid_pixels.float()) / 10.0            # (B,)  ~[0,1]
        features  = torch.cat([features, size_feat.unsqueeze(1)], dim=1)   # (B, feat_dim+1)
        return self.regressor(features)
