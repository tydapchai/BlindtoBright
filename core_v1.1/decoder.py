from enum import Enum

import numpy as np


class DecoderState(Enum):
    IDLE = 0
    SIGNING = 1


class TemporalDecoder:
    def __init__(self, idx_to_class, confidence_threshold=0.5, motion_threshold=0.005):
        self.idx_to_class = idx_to_class
        self.confidence_threshold = confidence_threshold
        self.motion_threshold = motion_threshold
        self.reset()

    def reset(self):
        self.state = DecoderState.IDLE
        self.smoothed_probs = None
        self.sentence_buffer = []
        self.idle_frames = 0
        self.last_word = None

    def process(self, probabilities, motion, hand_detected):
        if self.smoothed_probs is None:
            self.smoothed_probs = probabilities
        else:
            self.smoothed_probs = 0.5 * probabilities + 0.5 * self.smoothed_probs
        best_index = int(np.argmax(self.smoothed_probs))
        confidence = float(self.smoothed_probs[best_index])
        word = self.idx_to_class.get(best_index, "")
        moving = motion > self.motion_threshold
        confident = confidence > self.confidence_threshold

        if self.state == DecoderState.IDLE:
            if moving or (confident and hand_detected):
                self.state = DecoderState.SIGNING
                self.idle_frames = 0
            return None, None

        new_word = None
        if confident and hand_detected and word and word != self.last_word:
            self.sentence_buffer.append(word)
            self.last_word = word
            new_word = word
        if not hand_detected or (not moving and not confident):
            self.idle_frames += 1
        else:
            self.idle_frames = 0
        if self.idle_frames >= 15:
            sentence = self.sentence_buffer.copy() or None
            self.reset()
            return new_word, sentence
        return new_word, None