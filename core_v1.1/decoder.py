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


IDLE_WORDS = {"idle", "null", "none", "background", "khong_cu_chi", "nghi", "nghỉ", ""}


def is_idle_word(word: str) -> bool:
    """Kiểm tra xem nhãn có phải là trạng thái nghỉ/nhiễu nền hay không."""
    if not word:
        return True
    return word.strip().lower() in IDLE_WORDS


class ContinuousWordSpotter:
    """
    Thuật toán phát hiện và chốt từ liên tục (Continuous Sign Language Recognition)
    dựa trên kỹ thuật Dynamic Hysteresis & Peak Detection (Tìm đỉnh xác suất) trên cửa sổ trượt.

    Cơ chế nâng cấp:
    1. Lọc bỏ hoàn toàn trạng thái IDLE / nghỉ / đứng im: KHÔNG BAO GIỜ thêm 'Idle' vào câu.
    2. Bộ lọc đỉnh xác suất (min_peak_conf): Chỉ chốt từ khi cử chỉ đạt đỉnh đủ rõ ràng (>= 0.42),
       loại bỏ hoàn toàn các động tác lướt qua hoặc nhiễu chuyển tay.
    3. Nhận diện kết thúc cử chỉ tự nhiên qua Peak-Drop (xác suất tụt >= 32% từ đỉnh).
    4. Làm mượt xác suất (EMA) giúp khử rung giật frame đơn lẻ.
    """
    def __init__(
        self,
        idx_to_class,
        confidence_threshold=0.35,
        margin_threshold=0.08,
        min_peak_conf=0.42,
        min_hold_steps=2,
        cooldown_steps=5,
        min_motion=0.0030,
        peak_drop_ratio=0.68,
        ema_alpha=0.70,
    ):
        self.idx_to_class = idx_to_class
        self.conf_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.min_peak_conf = min_peak_conf
        self.min_hold_steps = min_hold_steps
        self.cooldown_steps = cooldown_steps
        self.min_motion = min_motion
        self.peak_drop_ratio = peak_drop_ratio
        self.ema_alpha = ema_alpha
        self.reset()

    def reset(self):
        self.candidate_word = None
        self.candidate_peak_conf = 0.0
        self.candidate_peak_margin = 0.0
        self.hold_count = 0
        self.words_sequence = []
        self.cooldown_counter = 0
        self.last_committed_word = None
        self.last_committed_time = 0.0
        self.smoothed_probs = None

    def get_top_k(self, probabilities, k=5):
        if probabilities is None or len(probabilities) == 0:
            return [("", 0.0)]
        top_indices = np.argsort(probabilities)[-k:][::-1]
        return [(self.idx_to_class.get(int(idx), f"Class_{idx}"), float(probabilities[idx])) for idx in top_indices]

    def update(self, probabilities, hand_detected=True, motion=0.0):
        """
        Cập nhật mỗi bước suy luận từ cửa sổ trượt.
        Trả về:
            committed_word: Từ vừa được chốt (nếu có, nếu không thì None)
            candidate_word: Từ đang theo dõi trực tiếp (bỏ qua nếu là Idle)
            candidate_conf: Xác suất đỉnh của từ đang theo dõi
            top5: Danh sách Top 5 nhãn hiện tại
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Làm mượt xác suất qua thời gian (Exponential Moving Average)
        raw_probs = np.array(probabilities, dtype=np.float32)
        if self.smoothed_probs is None or len(self.smoothed_probs) != len(raw_probs):
            self.smoothed_probs = raw_probs
        else:
            self.smoothed_probs = self.ema_alpha * raw_probs + (1.0 - self.ema_alpha) * self.smoothed_probs

        top5 = self.get_top_k(self.smoothed_probs, k=5)
        top1_w, top1_c = top5[0]
        top2_w, top2_c = top5[1] if len(top5) > 1 else ("", 0.0)
        margin = top1_c - top2_c

        # Kiểm tra trạng thái Idle (qua tên nhãn model hoặc khi không có tay)
        is_idle_class = is_idle_word(top1_w)

        # Cử chỉ hợp lệ: có tay, không phải class Idle, xác suất & margin đạt chuẩn
        # QUAN TRỌNG: Không phạt vận tốc chuyển động (motion) vì khi làm cử chỉ tĩnh (pose giữ yên),
        # tay sẽ đứng im nhưng xác suất mô hình lại đạt đỉnh cao nhất!
        is_valid_sign = (
            hand_detected
            and (not is_idle_class)
            and (top1_c >= self.conf_threshold)
            and (margin >= self.margin_threshold)
        )

        committed_word = None

        if is_valid_sign:
            if self.candidate_word == top1_w:
                # Đang duy trì cùng một cử chỉ:
                self.hold_count += 1
                if top1_c > self.candidate_peak_conf:
                    self.candidate_peak_conf = top1_c
                    self.candidate_peak_margin = margin

                # Điều kiện 1: CHỐT NGAY KHI GIỮ TAY VỮNG (Sustained Hold)
                # Khi người dùng làm cử chỉ và giữ vững >= min_hold_steps với xác suất cao (>= min_peak_conf):
                # -> Chốt ngay lập tức vào danh sách, không bắt người dùng phải hạ tay hay đợi tụt %!
                if self.hold_count >= self.min_hold_steps and self.candidate_peak_conf >= self.min_peak_conf:
                    committed_word = self._commit_word(self.candidate_word)

                # Điều kiện 2: CHỐT KHI SỤT ĐỈNH (Peak-Drop) nếu là cử chỉ chuyển động nhanh
                elif self.hold_count >= 1 and self.candidate_peak_conf >= self.min_peak_conf:
                    if top1_c <= self.candidate_peak_conf * self.peak_drop_ratio:
                        committed_word = self._commit_word(self.candidate_word)
            else:
                # Một từ mới hợp lệ xuất hiện và chiếm ưu thế:
                # Nếu từ cũ đã đạt đỉnh chuẩn (>= min_peak_conf) -> Chốt từ cũ trước!
                if (
                    self.candidate_word is not None
                    and self.hold_count >= 1
                    and self.candidate_peak_conf >= self.min_peak_conf
                ):
                    committed_word = self._commit_word(self.candidate_word)

                # Chuyển sang theo dõi từ mới (nếu không trùng với từ vừa chốt trong giai đoạn cooldown)
                if top1_w != self.last_committed_word or self.cooldown_counter == 0:
                    self.candidate_word = top1_w
                    self.candidate_peak_conf = top1_c
                    self.candidate_peak_margin = margin
                    self.hold_count = 1
                else:
                    self.candidate_word = None
                    self.candidate_peak_conf = 0.0
                    self.hold_count = 0
        else:
            # Rơi vào trạng thái Idle hoặc giai đoạn chuyển tay giữa 2 từ:
            # Nếu cử chỉ trước đó đã đạt đỉnh chuẩn (>= min_peak_conf) -> Chốt cử chỉ đó ngay khi hạ tay!
            if (
                self.candidate_word is not None
                and self.hold_count >= 1
                and self.candidate_peak_conf >= self.min_peak_conf
            ):
                committed_word = self._commit_word(self.candidate_word)

            # Reset trạng thái theo dõi khi ở trạng thái Idle, tuyệt đối không lưu Idle
            self.candidate_word = None
            self.candidate_peak_conf = 0.0
            self.hold_count = 0

        # Nếu candidate_word là Idle thì ẩn đi, không hiển thị trên giao diện
        display_candidate = None if is_idle_word(self.candidate_word) else self.candidate_word

        return committed_word, display_candidate, self.candidate_peak_conf, top5

    def _commit_word(self, word):
        """Chốt từ vào chuỗi câu. TUYỆT ĐỐI BỎ QUA nếu là trạng thái IDLE."""
        if not word or is_idle_word(word):
            self.candidate_word = None
            self.candidate_peak_conf = 0.0
            self.hold_count = 0
            return None

        import time
        self.words_sequence.append(word)
        self.last_committed_word = word
        self.last_committed_time = time.time()
        self.cooldown_counter = self.cooldown_steps
        self.candidate_word = None
        self.candidate_peak_conf = 0.0
        self.hold_count = 0
        return word

    def remove_last_word(self):
        if self.words_sequence:
            removed = self.words_sequence.pop()
            self.last_committed_word = self.words_sequence[-1] if self.words_sequence else None
            return removed
        return None

    def clear(self):
        self.reset()