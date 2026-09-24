import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stgcn_model import load_stgcn_checkpoint


checkpoint_path = CURRENT_DIR.parent / "models" / "best_vsl_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, num_classes = load_stgcn_checkpoint(checkpoint_path, device)
with torch.no_grad():
    output = model(torch.zeros(1, 9, 48, 76, device=device))
assert tuple(output.shape) == (1, num_classes)
print(f"OK: input=(1, 9, 48, 76), output={tuple(output.shape)}, device={device}")