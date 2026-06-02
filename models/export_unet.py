"""
TinyUNet for PBF neural post-processing - export to ONNX.

Architecture (≈230K parameters):
  Encoder: 3 strided-conv stages (4 → 16 → 32 → 64 channels)
  Bottleneck: Conv(64→64)
  Decoder: 3 ConvTranspose2d stages with skip connections
  Output: Conv(16→3) + identity residual (model learns a residual over the input RGB)

Input:  frame_rgba  [1, 4, H, W]  float32, values in [0, 1]
Output: enhanced_rgb [1, 3, H, W]  float32, values in [0, 1]

The model is initialised with near-identity weights so the untrained output
looks reasonable (≈ input) out of the box. Fine-tune with your own data to
add denoising / detail-enhancement behaviour.

Usage:
    pip install torch torchvision onnx onnxruntime
    python models/export_unet.py           # saves to Bin models unet_postprocess.onnx
    python models/export_unet.py --out path/to/output.onnx
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvRelu(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )


class TinyUNet(nn.Module):
    """
    Lightweight U-Net for real-time frame post-processing.
    All operations supported by DirectML (Conv2d, ConvTranspose2d, ReLU, Add).
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvRelu(4,  16)          # [1, 16, H,   W  ]
        self.enc2 = ConvRelu(16, 32, stride=2) # [1, 32, H/2, W/2]
        self.enc3 = ConvRelu(32, 64, stride=2) # [1, 64, H/4, W/4]

        # Bottleneck
        self.bottleneck = ConvRelu(64, 64)

        # Decoder (ConvTranspose2d for 2x up, then conv to merge skip)
        self.up3   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec3  = ConvRelu(64, 32)   # 32 up + 32 enc2 skip

        self.up2   = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2  = ConvRelu(32, 16)   # 16 up + 16 enc1 skip

        # Output: 1×1 conv → 3 channels (RGB), plus residual from input RGB
        self.head  = nn.Conv2d(16, 3, 1)

        self._init_weights()

    def _init_weights(self):
        # Tiny random init (not zero) so constant-folding can't eliminate the network.
        # The small scale keeps untrained output close to the input residual.
        nn.init.normal_(self.head.weight, std=0.5)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, H, W]
        H, W = x.shape[2], x.shape[3]

        # Pad H and W to multiples of 4 so the stride-2 encoder and ConvTranspose2d decoder align.
        # Without this, floor(H/2)*2 != H when H is not divisible by 4, causing a skip-connection
        # shape mismatch at the Concat. Padding is baked in as a static op at export time.
        pH = (4 - H % 4) % 4
        pW = (4 - W % 4) % 4
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH))

        s1 = self.enc1(x)          # [B, 16, H',   W'  ]
        s2 = self.enc2(s1)         # [B, 32, H'/2, W'/2]
        s3 = self.enc3(s2)         # [B, 64, H'/4, W'/4]
        b  = self.bottleneck(s3)   # [B, 64, H'/4, W'/4]

        d = self.dec3(torch.cat([self.up3(b), s2], dim=1))
        d = self.dec2(torch.cat([self.up2(d), s1], dim=1))

        rgb_in = x[:, :3]
        out = torch.clamp(rgb_in + self.head(d), 0.0, 1.0)

        # Crop any padding back to the original spatial size.
        return out[:, :, :H, :W]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(out_path: str, height: int = 1080, width: int = 1920):
    model = TinyUNet().eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"TinyUNet  —  {param_count:,} parameters")

    dummy = torch.zeros(1, 4, height, width)

    # Validate forward pass
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 3, height, width), f"Unexpected output shape: {out.shape}"
    print(f"Forward pass OK  —  output shape: {tuple(out.shape)}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # No dynamic_axes: DML EP's Concat (Join) operator requires fully resolved strides,
    # which dynamic H/W axes prevent. Export at fixed resolution and regenerate the
    # .onnx if the render resolution changes.
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["frame_rgba"],
        output_names=["enhanced_rgb"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Exported  →  {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "Bin", "models", "unet_postprocess.onnx"))
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width",  type=int, default=1920)
    args = parser.parse_args()
    export(os.path.normpath(args.out), args.height, args.width)
