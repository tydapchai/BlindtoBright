from enum import Enum

import numpy as np


class DecoderState(Enum):
    IDLE = 0
    SIGNING = 1


class TemporalDecoder:
    def __init__(
        self,
        idx_to_class,
        confidence_threshold=0.30,
        margin_threshold=0.08,
        motion_threshold=0.004,
        max_idle_steps=8,
    ):
        self.idx_to_class = idx_to_class
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.motion_threshold = motion_threshold
        self.max_idle_steps = max_idle_steps
        self.reset()

    def reset(self):
        self.state = DecoderState.IDLE
        self.smoothed_probs = None
        self.sentence_buffer = []
        self.idle_frames = 0
        self.last_word = None
        self.last_word_time = 0.0

    def get_top_k(self, probabilities, k=3):
        """Lấy danh sách Top-K nhãn từ vựng có xác suất cao nhất."""
        if probabilities is None or len(probabilities) == 0:
            return [("", 0.0)]
        top_indices = np.argsort(probabilities)[-k:][::-1]
        results = []
        for idx in top_indices:
            word = self.idx_to_class.get(int(idx), f"Class_{idx}")
            conf = float(probabilities[idx])
            results.append((word, conf))
        return results

    def decode_segment(self, probabilities, hand_detected=True):
        """
        Giải mã một cử chỉ trọn vẹn (Segmented Gesture) sau khi đã resample về 48 frame.
        Áp dụng bộ lọc Margin (Top-1 - Top-2 >= margin_threshold) để loại bỏ nhầm lẫn giữa các từ gần giống.
        Trả về: (new_word, is_valid, top3_candidates)
        """
        top3 = self.get_top_k(probabilities, k=3)
        top1_word, top1_conf = top3[0]
        top2_word, top2_conf = top3[1] if len(top3) > 1 else ("", 0.0)
        margin = top1_conf - top2_conf

        is_confident = top1_conf >= self.confidence_threshold
        is_distinct = margin >= self.margin_threshold
        is_valid = is_confident and is_distinct and hand_detected

        import time
        now = time.time()
        new_word = None
        if is_valid and top1_word:
            # Cho phép thêm từ nếu là từ khác hoặc nếu là cùng 1 từ nhưng cách nhau > 2.0s
            if (top1_word != self.last_word) or (now - self.last_word_time > 2.0):
                self.sentence_buffer.append(top1_word)
                self.last_word = top1_word
                self.last_word_time = now
                new_word = top1_word
                self.idle_frames = 0

        return new_word, is_valid, top3

    def commit_sentence(self):
        """Chốt câu thủ công: lấy danh sách các từ đã ghép và xóa bộ đệm câu."""
        if not self.sentence_buffer:
            return None
        sentence = self.sentence_buffer.copy()
        self.sentence_buffer.clear()
        self.last_word = None
        self.idle_frames = 0
        return sentence

    def remove_last_word(self):
        """Xóa từ cuối cùng vừa nhận diện nếu phát hiện bị nhận nhầm."""
        if self.sentence_buffer:
            removed = self.sentence_buffer.pop()
            self.last_word = self.sentence_buffer[-1] if self.sentence_buffer else None
            return removed
        return None

    def clear_sentence(self):
        """Xóa toàn bộ câu đang ghép để làm lại câu mới."""
        self.sentence_buffer.clear()
        self.last_word = None
        self.idle_frames = 0

    def step_idle(self):
        """Tăng số bước nghỉ. Khi nghỉ đủ lâu, chốt câu và trả về danh sách từ (cho chế độ auto cũ nếu cần)."""
        self.idle_frames += 1
        if self.idle_frames >= self.max_idle_steps and self.sentence_buffer:
            sentence = self.sentence_buffer.copy()
            self.sentence_buffer.clear()
            self.last_word = None
            self.idle_frames = 0
            return sentence
        return None

    def process(self, probabilities, motion, hand_detected):
        """Hỗ trợ chế độ stream liên tục (backward compatibility)."""
        if self.smoothed_probs is None:
            self.smoothed_probs = probabilities
        else:
            self.smoothed_probs = 0.4 * probabilities + 0.6 * self.smoothed_probs

        top3 = self.get_top_k(self.smoothed_probs, k=3)
        top1_word, top1_conf = top3[0]
        top2_word, top2_conf = top3[1] if len(top3) > 1 else ("", 0.0)
        margin = top1_conf - top2_conf

        moving = motion > self.motion_threshold
        confident = (top1_conf >= self.confidence_threshold) and (margin >= self.margin_threshold)

        if self.state == DecoderState.IDLE:
            if moving or (confident and hand_detected):
                self.state = DecoderState.SIGNING
                self.idle_frames = 0

        new_word = None
        if self.state == DecoderState.SIGNING:
            if confident and hand_detected and top1_word and top1_word != self.last_word:
                self.sentence_buffer.append(top1_word)
                self.last_word = top1_word
                new_word = top1_word
                self.idle_frames = 0

            if not hand_detected or (not moving and not confident):
                self.idle_frames += 1
            else:
                self.idle_frames = 0

            if self.idle_frames >= self.max_idle_steps:
                sentence = self.sentence_buffer.copy() or None
                self.reset()
                return new_word, sentence, top1_word, top1_conf, top3

        return new_word, None, top1_word, top1_conf, top3