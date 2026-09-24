import torch
import torch.nn as nn


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes=76, temporal_kernel=9):
        super().__init__()
        self.A = nn.Parameter(torch.ones((num_nodes, num_nodes)) / num_nodes)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        padding = (temporal_kernel - 1) // 2
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(padding, 0)
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.spatial_conv(x)
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.temporal_conv(x)
        x = self.bn(x)
        return self.relu(x + residual)


class STGCNTransformer(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=128,
                 num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.stgcn = nn.Sequential(
            STGCNBlock(in_channels, 64, num_nodes),
            STGCNBlock(64, d_model, num_nodes),
        )
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2,
            batch_first=True, dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.stgcn(x).mean(dim=-1).permute(0, 2, 1)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x).mean(dim=1)
        return self.classifier(x)


def load_stgcn_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {
        key: value for key, value in checkpoint["model_state_dict"].items()
        if key not in {"total_ops", "total_params"}
        and not key.endswith((".total_ops", ".total_params"))
    }
    num_classes = state_dict["classifier.weight"].shape[0]
    model = STGCNTransformer(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, checkpoint, num_classes