from collections import deque

import numpy as np
import torch


SEQUENCE_LENGTH = 48
NUM_NODES = 76


def extract_landmarks(results):
    landmarks = np.zeros((NUM_NODES, 3), dtype=np.float32)
    if results.pose_landmarks:
        for index in range(33):
            point = results.pose_landmarks.landmark[index]
            landmarks[index] = [point.x, point.y, point.z]
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]
        landmarks[75] = [
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2,
        ]
    if results.left_hand_landmarks:
        for index in range(21):
            point = results.left_hand_landmarks.landmark[index]
            landmarks[33 + index] = [point.x, point.y, point.z]
    if results.right_hand_landmarks:
        for index in range(21):
            point = results.right_hand_landmarks.landmark[index]
            landmarks[54 + index] = [point.x, point.y, point.z]
    return landmarks


def impute_missing_landmarks(sequence):
    """
    Bù đắp các landmark bàn tay bị MediaPipe bỏ sót tạm thời giữa các frame.
    Sử dụng forward-fill và backward-fill để tránh bước nhảy tọa độ giả gây spike vận tốc.
    """
    seq = np.asarray(sequence, dtype=np.float32).copy()
    T = seq.shape[0]
    for start_idx, end_idx in [(33, 54), (54, 75)]:
        has_hand = [bool(np.any(seq[t, start_idx:end_idx] != 0)) for t in range(T)]
        if not any(has_hand):
            continue  # Cả chuỗi không xuất hiện bàn tay này

        # Forward fill
        last_valid = None
        for t in range(T):
            if has_hand[t]:
                last_valid = seq[t, start_idx:end_idx].copy()
            elif last_valid is not None:
                seq[t, start_idx:end_idx] = last_valid.copy()

        # Backward fill cho các frame đầu
        last_valid = None
        for t in range(T - 1, -1, -1):
            if np.any(seq[t, start_idx:end_idx] != 0):
                last_valid = seq[t, start_idx:end_idx].copy()
            elif last_valid is not None:
                seq[t, start_idx:end_idx] = last_valid.copy()
    return seq


def resample_sequence(sequence, target_len=SEQUENCE_LENGTH):
    """
    Nội suy co giãn một chuỗi gồm N frames bất kỳ về đúng target_len (48 frames),
    đảm bảo đặc trưng thời gian khớp chính xác với tập huấn luyện.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    N, num_nodes, num_coords = seq.shape
    if N == target_len:
        return seq
    if N <= 1:
        return np.repeat(seq, target_len, axis=0)

    old_time = np.linspace(0.0, 1.0, N)
    new_time = np.linspace(0.0, 1.0, target_len)
    flat = seq.reshape(N, num_nodes * num_coords)
    resampled = np.empty((target_len, num_nodes * num_coords), dtype=np.float32)
    for j in range(num_nodes * num_coords):
        resampled[:, j] = np.interp(new_time, old_time, flat[:, j])
    return resampled.reshape(target_len, num_nodes, num_coords)


def build_tensor(frame_buffer, device):
    """
    Chuyển đổi chuỗi frames thành tensor 9 kênh (Tọa độ, Vận tốc, Gia tốc) với shape (1, 9, 48, 76).
    Tự động bù khuyết landmark và co giãn về 48 frame nếu chuỗi có độ dài khác 48.
    """
    data = np.stack(frame_buffer, axis=0).astype(np.float32)
    if data.shape[0] != SEQUENCE_LENGTH:
        data = resample_sequence(data, target_len=SEQUENCE_LENGTH)

    data = impute_missing_landmarks(data)
    data = data - data[:, 75:76, :]

    velocity = np.zeros_like(data)
    velocity[1:] = data[1:] - data[:-1]
    velocity[0] = velocity[1]

    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    acceleration[0] = acceleration[1]

    combined = np.concatenate([data, velocity, acceleration], axis=-1)
    tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def motion_energy(current, previous):
    hand_motion = float(np.mean(np.abs(current[33:75] - previous[33:75])))
    arm_motion = float(np.mean(np.abs(current[11:23] - previous[11:23])))
    return max(hand_motion, arm_motion)


def new_buffer():
    return deque(maxlen=SEQUENCE_LENGTH)