import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CTRGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes=76, temporal_kernel=9):
        super().__init__()
        self.A_static = nn.Parameter(torch.ones((num_nodes, num_nodes)) / num_nodes, requires_grad=True)
        
        self.conv_q = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
        
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        padding = (temporal_kernel - 1) // 2
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(padding, 0))
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else lambda x: x

    def forward(self, x):
        res = self.residual(x)
        q = self.conv_q(x).mean(dim=2)
        k = self.conv_k(x).mean(dim=2)
        
        A_dynamic = torch.einsum('bcv,bcw->bvw', q, k)
        A_dynamic = torch.clamp(A_dynamic, min=-10.0, max=10.0) 
        A_dynamic = torch.softmax(A_dynamic, dim=-1)
        
        x_spatial = self.spatial_conv(x)
        x_static = torch.einsum('bctv,vw->bctw', x_spatial, self.A_static)
        x_dynamic = torch.einsum('bctv,bvw->bctw', x_spatial, A_dynamic)
        x = x_static + x_dynamic
        
        x = self.temporal_conv(x)
        x = self.bn(x)
        return self.relu(x + res)

class CTR_GCN_Model(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=128):
        super().__init__()
        self.ctr_gcn = nn.Sequential(
            CTRGCN_Block(in_channels, 64, num_nodes),
            CTRGCN_Block(64, d_model, num_nodes),
            CTRGCN_Block(d_model, d_model, num_nodes),
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, labels=None):
        x = self.ctr_gcn(x).mean(dim=-1).mean(dim=-1)
        return self.classifier(x)


def load_ctr_gcn_checkpoint(checkpoint_path, device):
    """Tải checkpoint CTR-GCN và tự động suy luận kiến trúc từ state_dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_sd = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {
        key: value
        for key, value in raw_sd.items()
        if key not in {"total_ops", "total_params"}
        and not key.endswith((".total_ops", ".total_params"))
    }

    num_classes = state_dict["classifier.weight"].shape[0]
    d_model = state_dict["classifier.weight"].shape[1]
    in_channels = (
        state_dict["ctr_gcn.0.spatial_conv.weight"].shape[1]
        if "ctr_gcn.0.spatial_conv.weight" in state_dict
        else 9
    )
    num_nodes = (
        state_dict["ctr_gcn.0.A_static"].shape[0]
        if "ctr_gcn.0.A_static" in state_dict
        else 76
    )

    model = CTR_GCN_Model(
        num_classes=num_classes,
        in_channels=in_channels,
        num_nodes=num_nodes,
        d_model=d_model,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, checkpoint, num_classes


def load_model_checkpoint(checkpoint_path, device):
    """
    Tự động nhận diện cấu trúc checkpoint (CTR-GCN hoặc ST-GCN Transformer)
    và nạp đúng mô hình.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_sd = checkpoint.get("model_state_dict", checkpoint)

    has_ctr_gcn = any("ctr_gcn" in k for k in raw_sd.keys())
    has_stgcn = any("stgcn" in k for k in raw_sd.keys())

    if has_ctr_gcn or "ctr_gcn" in str(checkpoint_path).lower():
        return load_ctr_gcn_checkpoint(checkpoint_path, device)
    elif has_stgcn or "stgcn" in str(checkpoint_path).lower():
        from stgcn_model import load_stgcn_checkpoint
        return load_stgcn_checkpoint(checkpoint_path, device)
    else:
        try:
            return load_ctr_gcn_checkpoint(checkpoint_path, device)
        except Exception:
            from stgcn_model import load_stgcn_checkpoint
            return load_stgcn_checkpoint(checkpoint_path, device)