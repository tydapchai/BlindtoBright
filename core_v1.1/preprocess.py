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


def build_tensor(frame_buffer, device):
    data = np.stack(frame_buffer, axis=0).astype(np.float32)
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
    return float(np.mean(np.abs(current[33:75] - previous[33:75])))


def new_buffer():
    return deque(maxlen=SEQUENCE_LENGTH)