import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ctr_gcn_model import load_ctr_gcn_checkpoint, load_model_checkpoint
from stgcn_model import load_stgcn_checkpoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Kiểm tra checkpoint CTR-GCN chính
ctr_gcn_path = CURRENT_DIR.parent / "models" / "best_vsl_model_ctr_gcn.pth"
if ctr_gcn_path.exists():
    model_ctr, ckpt_ctr, num_classes_ctr = load_ctr_gcn_checkpoint(ctr_gcn_path, device)
    with torch.no_grad():
        output_ctr = model_ctr(torch.zeros(1, 9, 48, 76, device=device))
    assert tuple(output_ctr.shape) == (1, num_classes_ctr)
    print(f"OK [CTR-GCN]: input=(1, 9, 48, 76), output={tuple(output_ctr.shape)}, classes={num_classes_ctr}, device={device}")

# 2. Kiểm tra checkpoint ST-GCN cũ (nếu có)
stgcn_path = CURRENT_DIR.parent / "models" / "best_vsl_model.pth"
if stgcn_path.exists():
    model_st, ckpt_st, num_classes_st = load_model_checkpoint(stgcn_path, device)
    with torch.no_grad():
        output_st = model_st(torch.zeros(1, 9, 48, 76, device=device))
    assert tuple(output_st.shape) == (1, num_classes_st)
    print(f"OK [ST-GCN]: input=(1, 9, 48, 76), output={tuple(output_st.shape)}, classes={num_classes_st}, device={device}")