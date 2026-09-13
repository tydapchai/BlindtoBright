import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

HAND_FEATURES = 63
RAW_FEATURES = 126
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17), (2, 5)
]

class GestureBiGRU(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, dropout=0.4, bidirectional=True):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, num_classes),
        )

    def forward(self, x):
        x = self.input_projection(x)
        seq, _ = self.gru(x)
        weights = torch.softmax(self.attention(seq).squeeze(-1), dim=1).unsqueeze(-1)
        return self.classifier(torch.sum(seq * weights, dim=1))

def normalize_hand(points):
    points = points.astype(np.float32).copy()
    points -= points[0]
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 1e-6:
        points /= scale
    return points

def frame_features(result, force_right_hand=True):
    output = np.zeros(RAW_FEATURES, dtype=np.float32)
    if not result.hand_landmarks: return output

    if force_right_hand and len(result.hand_landmarks) == 1:
        points = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        output[HAND_FEATURES:2*HAND_FEATURES] = normalize_hand(points).reshape(-1)
        return output

    used = set()
    for i, landmarks in enumerate(result.hand_landmarks):
        points = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        hand = result.handedness[i][0].category_name.lower() if result.handedness else None
        start = 0 if hand == "left" else HAND_FEATURES
        if hand not in {"left", "right"}: start = 0 if i == 0 else HAND_FEATURES
        if start in used: start = HAND_FEATURES if start == 0 else 0
        output[start:start+HAND_FEATURES] = normalize_hand(points).reshape(-1)
        used.add(start)
    return output

def resample(sequence, target):
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) == 1: return np.repeat(sequence, target, axis=0)
    old = np.linspace(0, 1, len(sequence))
    new = np.linspace(0, 1, target)
    result = np.empty((target, sequence.shape[1]), dtype=np.float32)
    for j in range(sequence.shape[1]):
        result[:, j] = np.interp(new, old, sequence[:, j])
    return result

def fill_missing_frames(sequence):
    if not sequence: return sequence
    filled = [arr.copy() for arr in sequence]
    last_valid = None
    for i in range(len(filled)):
        if np.any(filled[i] != 0): last_valid = filled[i]
        elif last_valid is not None: filled[i] = last_valid.copy()
    last_valid = None
    for i in range(len(filled) - 1, -1, -1):
        if np.any(filled[i] != 0): last_valid = filled[i]
        elif last_valid is not None: filled[i] = last_valid.copy()
    return filled

def prepare(sequence, length, mean, std):
    filled_sequence = fill_missing_frames(sequence)
    sequence = resample(filled_sequence, length)
    velocity = np.diff(sequence, axis=0, prepend=sequence[:1])
    sequence = np.concatenate([sequence, velocity], axis=1)
    return ((sequence - mean) / std).astype(np.float32)