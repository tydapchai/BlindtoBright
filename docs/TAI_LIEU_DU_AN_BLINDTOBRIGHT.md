# BÁO CÁO THUYẾT MINH DỰ ÁN
## CUỘC THI SÁNG TẠO CÔNG NGHỆ VÀ TRÍ TUỆ NHÂN TẠO · BẢNG C
### LĨNH VỰC: GIÁO DỤC (EDUCATION & ACCESSIBILITY)

---

# TÊN DỰ ÁN: BLINDTOBRIGHT (B2B)
### HỆ THỐNG TRỢ LÝ GIAO TIẾP HAI CHIỀU THỜI GIAN THỰC HỖ TRỢ NGƯỜI KHIẾM THÍNH VÀ KHIẾM THỊ DỰA TRÊN THỊ GIÁC MÁY TÍNH VÀ TRÍ TUỆ NHÂN TẠO ĐA PHƯƠNG THỨC

* **Tên đội thi / Tác giả:** Phạm Hoàng Minh & Cộng sự
* **Lĩnh vực đăng ký:** Giáo dục & Hỗ trợ Tiếp cận (EdTech & Social Accessibility)
* **Phiên bản tài liệu:** 1.2 (Kiến trúc ST-GCN Transformer v1.1 & Phần cứng nhúng ESP32)
* **Mã nguồn dự án:** https://github.com/tydapchai/BlindtoBright

---

## MỤC LỤC
1. [TÓM TẮT DỰ ÁN (EXECUTIVE SUMMARY)](#1-tóm-tắt-dự-án-executive-summary)
2. [BỐI CẢNH & TÍNH CẤP THIẾT CỦA ĐỀ TÀI](#2-bối-cảnh--tính-cấp-thiết-của-đề-tài)
3. [MỤC TIÊU DỰ ÁN & ĐỐI TƯỢNG HƯỚNG TỚI](#3-mục-tiêu-dự-án--đối-tượng-hướng-tới)
4. [KIẾN TRÚC KỸ THUẬT & NGUYÊN LÝ HOẠT ĐỘNG](#4-kiến-trúc-kỹ-thuật--nguyên-lý-hoạt-động)
   * 4.1. Luồng 1: Nhận diện thủ ngữ VSL sang tiếng nói (Sign-to-Speech)
   * 4.2. Luồng 2: Nhận diện giọng nói sang văn bản trợ giảng (Speech-to-Text)
   * 4.3. Thiết kế phần cứng biên nhúng ESP32
5. [ĐIỂM NỔI BẬT & ĐỔI MỚI SÁNG TẠO VỀ THUẬT TOÁN](#5-điểm-nổi-bật--đổi-mới-sáng-tạo-về-thuật-toán)
6. [KẾT QUẢ THỰC NGHIỆM & ĐÁNH GIÁ HIỆU NĂNG](#6-kết-quả-thực-nghiệm--đánh-giá-hiệu-năng)
7. [ỨNG DỤNG THỰC TẾ TRONG GIÁO DỤC & TÁC ĐỘNG XÃ HỘI](#7-ứng-dụng-thực-tế-trong-giáo-dục--tác-động-xã-hội)
8. [KÊ KHAI CÔNG CỤ AI, DỮ LIỆU, THƯ VIỆN & BẢN QUYỀN](#8-kê-khai-công-cụ-ai-dữ-liệu-thư-viện--bản-quyền)
9. [LỘ TRÌNH PHÁT TRIỂN & KẾT LUẬN](#9-lộ-trình-phát-triển--kết-luận)

---

## 1. TÓM TẮT DỰ ÁN (EXECUTIVE SUMMARY)

**BlindtoBright** là một giải pháp công nghệ giáo dục và trợ năng toàn diện, được thiết kế nhằm phá vỡ rào cản giao tiếp giữa người khiếm thính (sử dụng ngôn ngữ ký hiệu) và người bình thường (sử dụng ngôn ngữ nói) trong môi trường học đường và cuộc sống hàng ngày.

Khác với các hệ thống dịch thủ ngữ truyền thống đòi hỏi thiết bị đeo đắt tiền hoặc chỉ hoạt động ở dạng ảnh tĩnh một chiều, BlindtoBright là **hệ thống giao tiếp hai chiều thời gian thực (Real-time Bidirectional Assistant)** với chi phí phần cứng cực thấp (dưới 300.000 VNĐ):

* **Chiều 1 (Ký hiệu → Lời nói):** Sử dụng camera (Webcam hoặc module ESP32-CAM) trích xuất 76 điểm đặc trưng khung xương theo thời gian thực → đưa vào mạng nơ-ron tích chập đồ thị không - thời gian (**ST-GCN**) kết hợp **Transformer** để nhận diện 472 từ vựng Ngôn ngữ Ký hiệu Việt Nam (VSL) → sử dụng Mô hình Ngôn ngữ Lớn (**Gemini LLM**) tái cấu trúc ngữ pháp thành câu tiếng Việt hoàn chỉnh → phát âm thanh qua module loa phần cứng **ESP32 I2S DAC**.
* **Chiều 2 (Lời nói → Văn bản trợ giảng):** Giọng nói của giáo viên/bạn học được thu qua micro → bộ lọc **Silero VAD** và mô hình chuyển đổi giọng nói **Faster-Whisper** phiên âm tức thì thành văn bản trực quan trên màn hình Web/Điện thoại, tự động tóm tắt bài học giúp học sinh khiếm thính dễ dàng tiếp thu kiến thức.

---

## 2. BỐI CẢNH & TÍNH CẤP THIẾT CỦA ĐỀ TÀI

### 2.1. Thực trạng xã hội
Theo thống kê của Tổ chức Y tế Thế giới (WHO) và Tổng cục Thống kê Việt Nam, hiện có hơn **2,5 triệu người khuyết tật nghe - nói** tại Việt Nam. Trong môi trường giáo dục:
* Hơn **85% trẻ em khiếm thính** gặp khó khăn trong việc theo học tại các trường phổ thông hòa nhập do thiếu giáo viên chuyên trách ngôn ngữ ký hiệu.
* Giao tiếp giữa học sinh khiếm thính với thầy cô và bạn bè đồng trang lứa gần như bị cô lập, dẫn đến tỷ lệ bỏ học cao và hạn chế cơ hội tiếp cận tri thức đại học, nghề nghiệp.

### 2.2. Hạn chế của các giải pháp hiện nay
| Giải pháp hiện hành | Ưu điểm | Nhược điểm chí mạng |
| :--- | :--- | :--- |
| **Găng tay cảm biến (Data Glove)** | Bắt cử chỉ ngón tay tương đối chuẩn | Chi phí rất cao (> 5 - 10 triệu VNĐ), dễ đứt gãy dây cơ học, khó bảo quản, **không thể bắt được vị trí tay tương đối với khuôn mặt/ngực** (yếu tố quyết định nghĩa từ trong VSL). |
| **Phần mềm dịch ảnh tĩnh (Single-frame CNN)** | Nhẹ, dễ cài đặt | Chỉ nhận diện được bảng chữ cái ngón tay tĩnh (Fingerspelling), **hoàn toàn thất bại** với từ vựng động có tính biến thiên thời gian. |
| **Người phiên dịch thủ ngữ trực tiếp** | Chính xác cao | Chi phí đào tạo rất lớn, số lượng phiên dịch viên cực kỳ khan hiếm, không thể đáp ứng cho từng lớp học hay từng cá nhân. |

---

## 3. MỤC TIÊU DỰ ÁN & ĐỐI TƯỢNG HƯỚNG TỚI

* **Mục tiêu kỹ thuật:** Xây dựng thành công pipeline nhận diện ngôn ngữ ký hiệu liên tục (Continuous Sign Language Recognition) đạt độ trễ suy luận dưới 350 ms, xử lý trơn tru trên luồng video 25 - 30 FPS, tích hợp phần cứng nhúng phân tán ổn định.
* **Mục tiêu giáo dục:** Cung cấp "Người phiên dịch ảo" kiêm "Sổ tay trợ giảng thông minh" đặt ngay trên bàn học, giúp học sinh khiếm thính tự tin phát biểu, nghe giảng và thảo luận nhóm bình đẳng.
* **Đối tượng phục vụ:**
  1. Học sinh, sinh viên khiếm thính trong các lớp học hòa nhập và trường chuyên biệt.
  2. Giáo viên, giảng viên cần công cụ hỗ trợ truyền đạt kiến thức cho người học khiếm thính.
  3. Cộng đồng người yếu thế cần thiết bị giao tiếp y tế, hành chính công cộng.

---

## 4. KIẾN TRÚC KỸ THUẬT & NGUYÊN LÝ HOẠT ĐỘNG

Hệ thống được thiết kế theo mô hình kiến trúc lai (Hybrid Edge-Server Computing), tối ưu hóa luồng dữ liệu hai chiều:

<!-- PIPELINE_DIAGRAM_PLACEHOLDER -->

### 4.1. Luồng 1: Nhận diện thủ ngữ VSL sang tiếng nói (Sign-to-Speech Pipeline)

1. **Thu nhận hình ảnh:** Video được thu nhận qua Webcam máy tính hoặc luồng MJPEG stream độ trễ thấp từ module **ESP32-CAM** thông qua giao thức HTTP/WebSocket (`http://ESP_IP:81/stream`).
2. **Trích xuất đặc trưng hình thái (Skeletal Feature Extraction):**
   * Sử dụng framework **MediaPipe Holistic** trích xuất tọa độ không gian 3 chiều của **76 điểm khớp trọng yếu**:
     * 33 điểm cơ thể (Body Pose) → trích xuất tư thế thân trên, vai, khuỷu tay và cổ tay.
     * 42 điểm hai bàn tay (Left/Right Hand) → mỗi bàn tay 21 điểm đốt ngón.
     * 1 điểm mốc định vị đầu/mũi nhằm tính toán tọa độ tương đối.
   * Tất cả tọa độ được chuẩn hóa (Normalization) về hệ quy chiếu gốc tại điểm giữa hai vai (Mid-Shoulder), triệt tiêu sự sai khác về kích thước cơ thể và khoảng cách đứng xa/gần.
3. **Mô hình hóa động học 9 kênh (Kinematic Feature Engineering):**
   Thay vì chỉ sử dụng tọa độ vị trí tĩnh (x, y, z), hệ thống tính toán bổ sung:
   * Vector vận tốc: v(t) = p(t) - p(t-1)
   * Vector gia tốc: a(t) = v(t) - v(t-1)
   * Ma trận đầu vào có kích thước B × C × T × V = (B, 9, 48, 76), cho phép mô hình nắm bắt chính xác quán tính và tốc độ vung tay.
4. **Mạng nơ-ron đồ thị ST-GCN + Transformer:**
   * **Spatial GCN (Tích chập đồ thị không gian):** Học mối tương quan giải phẫu sinh học giữa các khớp ngón tay theo ma trận kề A.
   * **Temporal Convolution & Transformer (Nắm bắt thời gian):** Mô hình hóa chuỗi 48 khung hình liên tục để phân loại chính xác cử chỉ động.
5. **Bộ giải mã thời gian (Temporal Decoder & Margin Filter):**
   * Sử dụng kỹ thuật cửa sổ trượt (Sliding Window) với bước nhảy (`stride = 4`).
   * Tích hợp bộ lọc phân vân: chỉ chấp nhận từ khi xác suất vượt ngưỡng P > 0.65 và khoảng cách biên (Margin) giữa Top 1 và Top 2 vượt quá 8%, loại bỏ hoàn toàn các dự đoán nhiễu ở trạng thái nghỉ (Idle).
6. **Tái cấu trúc ngữ pháp tự nhiên qua LLM:**
   * Trong ngôn ngữ ký hiệu, ngữ pháp theo cấu trúc Topic-Comment (ví dụ: *['tôi', 'bài tập', 'không hiểu']*).
   * Chuỗi từ nhận diện được đưa vào **Gemini LLM** với Prompt hệ thống tối ưu hóa đặc thù ngữ nghĩa Việt Nam để chuyển thành: *"Em chưa hiểu bài tập này ạ."*
7. **Phát âm thanh ra phần cứng:**
   * Gemini TTS tổng hợp dữ liệu âm thanh dạng PCM Wave.
   * Gửi luồng raw audio qua HTTP POST `http://ESP_IP/play` tới ESP32 → giải mã qua giao tiếp I2S → khuếch đại qua chip MAX98357A và phát ra loa mini.

---

### 4.2. Luồng 2: Nhận diện giọng nói sang văn bản trợ giảng (Speech-to-Text Pipeline)

* Nhằm đảm bảo tương tác hai chiều, hệ thống cung cấp giao diện Web di động chạy trên nền tảng Flask/Waitress (`http://IP_HOST:8000`), có thể truy cập từ bất kỳ điện thoại hoặc máy tính bảng nào trong cùng mạng Wi-Fi lớp học.
* **Silero VAD (Voice Activity Detection):** Lọc bỏ tiếng ồn nền của lớp học (tiếng quạt, tiếng bước chân), chỉ kích hoạt luồng xử lý khi thực sự có tiếng nói của thầy cô.
* **Faster-Whisper Engine:** Mô hình Transformer được tối ưu hóa lượng tử hóa (int8 quantization), chạy mượt mà ngay trên CPU với độ trễ chuyển giọng nói thành văn bản dưới 600 ms.
* Văn bản hiển thị với kích thước chữ tùy biến cao, độ tương phản lớn, hỗ trợ chức năng lưu và tóm tắt nhanh nội dung bài giảng vào ứng dụng Ghi chú.

---

### 4.3. Thiết kế phần cứng biên nhúng ESP32

Hệ thống phần cứng được tối ưu theo tiêu chí nhỏ gọn, tiêu thụ điện năng thấp, dễ dàng gắn lên bàn học hoặc cặp sách:

<!-- HARDWARE_DIAGRAM_PLACEHOLDER -->

* **Sơ đồ kết nối chân I2S (Pinout ESP32 ↔ MAX98357A):**
  * `LRC (Word Select)` → `GPIO 25`
  * `BCLK (Bit Clock)` → `GPIO 26`
  * `DIN (Data In)` → `GPIO 22`
  * `VIN` → `5V`, `GND` → `GND`

---

## 5. ĐIỂM NỔI BẬT & ĐỔI MỚI SÁNG TẠO VỀ THUẬT TOÁN

1. **Giải quyết triệt để vấn đề mất thông tin ngón tay (Hand Shape):**
   * Các nghiên cứu trước đây thường dùng phương pháp ép trung bình sớm (Global Average Pooling) toàn bộ 76 khớp, khiến mô hình mất đi hình dạng gập ngón tay (vốn chiếm 80% ý nghĩa của từ).
   * BlindtoBright tách riêng nhánh **Hand Graph** (42 điểm) và **Body Pose** (33 điểm), giữ nguyên ma trận kề ngón tay xuyên suốt các tầng ST-GCN trước khi dung hợp (Fusion).
2. **Kỹ thuật Tăng cường Dữ liệu Động học (Online Kinematic Augmentation):**
   * Huấn luyện mô hình với cơ chế tự động xoay góc ngẫu nhiên ±12°, co giãn tỷ lệ 0.9 - 1.1, biến thiên tốc độ thời gian (Temporal Warping) và Joint Dropout 5%. Nhờ đó, hệ thống không bị "học vẹt" góc ngồi và duy trì độ chính xác cao khi người dùng đứng lệch camera.
3. **Cơ chế Hậu xử lý Ngữ pháp kết hợp AI Tạo sinh (GenAI Syntax Bridge):**
   * Là một trong những dự án tiên phong tích hợp LLM vào bài toán ngôn ngữ ký hiệu tiếng Việt, biến các từ rời rạc thành văn phong giao tiếp lễ phép, tự nhiên, phù hợp văn hóa ứng xử học đường Việt Nam.

---

## 6. KẾT QUẢ THỰC NGHIỆM & ĐÁNH GIÁ HIỆU NĂNG

### 6.1. Tập dữ liệu & Môi trường thực nghiệm
* **Bộ dữ liệu:** 472 lớp từ vựng VSL bao gồm các chủ đề học đường, chào hỏi, y tế, nhu cầu cá nhân.
* **Môi trường huấn luyện:** NVIDIA GPU trên môi trường PyTorch, cấu hình batch size 32, tối ưu hóa qua AdamW và Cosine Annealing Learning Rate.

### 6.2. Kết quả kiểm thử định lượng
| Tiêu chí đánh giá | Kết quả đạt được | Ý nghĩa thực tế |
| :--- | :--- | :--- |
| **Độ chính xác Top-1 (Validation Acc)** | **88.4%** (trên tập 472 nhãn VSL) | Nhận diện chính xác hầu hết từ vựng thông dụng |
| **Độ chính xác Top-3** | **96.1%** | Đảm bảo từ đúng luôn nằm trong vùng xem xét của decoder |
| **Thời gian trích xuất Keypoint (MediaPipe)** | 12 - 15 ms / frame | Đạt tốc độ 30 FPS thời gian thực |
| **Thời gian suy luận mô hình ST-GCN** | 18 ms / window (GPU) / 65 ms (CPU) | Chạy mượt mà trên laptop thông thường |
| **Độ trễ toàn trình (End-to-End Latency)** | **Dưới 380 ms** | Không có cảm giác khựng hay trễ khi nói chuyện |
| **Chi phí chế tạo phần cứng** | **Khoảng 280.000 VNĐ** | Rẻ gấp 20 - 30 lần thiết bị thương mại |

---

## 7. ỨNG DỤNG THỰC TẾ TRONG GIÁO DỤC & TÁC ĐỘNG XÃ HỘI

### 7.1. Ứng dụng trong lớp học hòa nhập
* **Trợ lý phát biểu:** Học sinh khiếm thính tự tin ra ký hiệu tại chỗ; hệ thống nhận diện và phát âm thanh to, rõ ràng qua loa ESP32 để giáo viên và cả lớp cùng nghe.
* **Bảng phụ đề trực tiếp:** Lời giảng của thầy cô được chuyển thành văn bản chạy trên màn hình của học sinh, giúp em không bị bỏ sót kiến thức bài học.

### 7.2. Ứng dụng trong tự học ngôn ngữ ký hiệu
* Hệ thống đóng vai trò như một **"Gia sư ảo"**: Học sinh bình thường hoặc giáo viên muốn học thủ ngữ có thể đứng trước camera thực hành, hệ thống sẽ chấm điểm và phản hồi xem động tác tay đã chuẩn xác hay chưa.

### 7.3. Ý nghĩa nhân văn & Mục tiêu phát triển bền vững (SDGs)
* Đóng góp trực tiếp vào **Mục tiêu SDG 4 (Giáo dục có chất lượng)** và **Mục tiêu SDG 10 (Giảm bất bình đẳng)** của Liên Hợp Quốc, mở ra cơ hội học tập bình đẳng cho cộng đồng người khuyết tật tại Việt Nam.

---

## 8. KÊ KHAI CÔNG CỤ AI, DỮ LIỆU, THƯ VIỆN & BẢN QUYỀN

Tuân thủ nghiêm ngặt quy chế cuộc thi và tính liêm chính học thuật:

1. **Công cụ Trí tuệ Nhân tạo & API:**
   * **Google Gemini API:** Sử dụng cho tác vụ tái cấu trúc ngữ pháp (Grammar Refinement) và tổng hợp giọng nói tự nhiên (TTS).
   * **Faster-Whisper & Silero VAD:** Sử dụng mô hình mã nguồn mở chuyển đổi giọng nói thành văn bản.
2. **Framework & Thư viện mã nguồn mở:**
   * `PyTorch`: Xây dựng và huấn luyện mô hình ST-GCN & Transformer.
   * `MediaPipe Holistic` (Google): Trích xuất tọa độ khớp xương.
   * `OpenCV`, `NumPy`: Xử lý luồng video và ma trận đặc trưng.
   * `Flask`, `Waitress`: Triển khai máy chủ Web trợ giảng nội bộ.
3. **Dữ liệu huấn luyện:**
   * Bộ dữ liệu VSL (Vietnamese Sign Language) được chuẩn hóa từ các nguồn mở học thuật và dữ liệu tự thu thập phục vụ mục đích nghiên cứu phi thương mại.
4. **Cam kết đạo đức & Quyền riêng tư:**
   * Hệ thống **chỉ xử lý các điểm tọa độ trừu tượng (Keypoints)**, tuyệt đối không lưu trữ hay truyền tải hình ảnh khuôn mặt người dùng lên máy chủ đám mây, đảm bảo 100% quyền riêng tư của học sinh.

---

## 9. LỘ TRÌNH PHÁT TRIỂN & KẾT LUẬN

### 9.1. Lộ trình phát triển sản phẩm (Roadmap)
* **Giai đoạn 1 (Hiện tại):** Hoàn thiện prototype hoạt động ổn định trên PC/Server kết nối thiết bị ngoại vi ESP32; hỗ trợ 472 từ vựng VSL.
* **Giai đoạn 2 (3 - 6 tháng tới):** 
  * Nén lượng tử hóa mô hình qua định dạng ONNX / TensorRT để triển khai trực tiếp lên máy tính nhúng biên độc lập (Raspberry Pi 5 hoặc Orange Pi AI).
  * Mở rộng từ điển lên 1.000+ từ vựng chuyên sâu cho các môn Toán, Khoa học Tự nhiên và Lịch sử.
* **Giai đoạn 3 (1 năm tới):** Thiết kế vỏ hộp công nghiệp in 3D hoàn chỉnh, đóng gói thành sản phẩm thương mại hóa giá rẻ hỗ trợ các trường học chuyên biệt trên toàn quốc.

### 9.2. Lời kết
**BlindtoBright** không chỉ dừng lại ở một bài thi công nghệ, mà là khát vọng mang lại "Ánh sáng của sự thấu hiểu" cho cộng đồng người khiếm thính. Dự án chứng minh rằng với sự kết hợp thông minh giữa các thuật toán AI hiện đại và phần cứng giá rẻ, chúng ta hoàn toàn có thể tạo ra những giải pháp công nghệ nhân ái, xóa nhòa khoảng cách và đem lại sự bình đẳng trong giáo dục.

---
*Hồ sơ dự án được lập và cam kết tính xác thực bởi Đội thi BlindtoBright.*
