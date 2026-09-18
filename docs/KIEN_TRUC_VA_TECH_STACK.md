# Kiến trúc và Tech Stack

## 1. Tổng quan

HandSign là ứng dụng Python chạy local bằng command line, dùng để:

- Nhận diện từ ngôn ngữ ký hiệu trong video.
- So sánh động tác của người dùng với mẫu chuẩn.
- Chấm điểm từng tay, khuôn mặt và tạo feedback theo từng đoạn động tác.

Hệ thống hiện không có frontend, backend API hoặc database.

## 2. Kiến trúc hệ thống

```text
Webcam / Video
    |
    +-- MediaPipe --> Landmark 2D --> Trim + Normalize
    |                                      |
    |                                      v
    |                              FastDTW với mẫu chuẩn
    |                                      |
    |                                      v
    |                         Điểm vùng + Feedback lỗi
    |
    +-- VideoMAE --> Nhãn dự đoán + độ tin cậy
                                           |
                                           v
                              data/user/evaluation.json
```

Hệ thống được chia thành ba phần độc lập:

1. **Tracking:** MediaPipe lấy landmark của hai tay và khuôn mặt.
2. **Recognition:** VideoMAE dự đoán người dùng đang thực hiện từ nào.
3. **Form evaluation:** FastDTW so sánh landmark với mẫu chuẩn và chấm chất lượng động tác.

## 3. Các thành phần chính

| Thành phần | File | Trách nhiệm |
| --- | --- | --- |
| Thu thập dữ liệu | `record.py` | Quay webcam, lấy landmark, cắt đoạn động tác và lưu NPY/NPZ |
| Nhận diện từ | `recognizer.py` | Lấy 16 frame và phân loại video bằng VideoMAE |
| Điều phối đánh giá | `evaluate.py` | Gọi recognition, form evaluation và ghi kết quả JSON |
| Xử lý landmark | `sign_eval/landmarks.py` | Validate, normalize, kiểm tra tracking và tính DTW |
| Chấm điểm | `sign_eval/evaluator.py` | Tổng hợp điểm vùng, trạng thái và feedback |
| Calibration | `sign_eval/calibration.py` | Tạo ngưỡng riêng cho từng nhãn và từng vùng cơ thể |
| Validation | `sign_eval/validation.py` | Fit ngưỡng từ các clip được gán nhãn đúng/sai |

## 4. Dữ liệu landmark

Mỗi frame có 48 điểm 2D:

- Tay trái: 21 điểm.
- Tay phải: 21 điểm.
- Khuôn mặt: 6 điểm.

Dữ liệu được chuẩn hóa bằng z-score và so sánh theo ba vùng: `left_hand`, `right_hand`, `face`.

## 5. Cách chấm điểm

- FastDTW căn chỉnh hai chuỗi động tác có tốc độ hoặc số frame khác nhau.
- Khoảng cách được tính riêng cho từng vùng cơ thể.
- Ngưỡng vùng được tạo trước từ tập mẫu chuẩn và lưu trong `artifacts/label_calibration.json`.
- Tay chiếm tổng cộng 80% và khuôn mặt chiếm 20% điểm. Nếu không chấm khuôn mặt, hai tay chia nhau 100%.
- Kết quả cuối được lưu tại `data/user/evaluation.json`.

## 6. Tech stack

| Công nghệ | Mục đích |
| --- | --- |
| Python | Ngôn ngữ chính và CLI |
| NumPy | Xử lý mảng landmark và tính toán số học |
| OpenCV | Quay, hiển thị, đọc và ghi video |
| MediaPipe Tasks | Phát hiện landmark tay và khuôn mặt |
| FastDTW | So sánh chuỗi động tác |
| PyTorch | Chạy mô hình deep learning |
| Hugging Face Transformers | Tải và chạy VideoMAE |
| Hugging Face Hub | Tải checkpoint của mô hình |
| Decord | Đọc và lấy frame video |
| unittest | Kiểm thử tự động |
| JSON, NPY, NPZ | Lưu calibration, landmark và kết quả |

Mô hình nhận diện đang dùng: `star092304/vi-sign-language-videomae-base`.

## 7. Kiểu triển khai

Đây là **modular monolith chạy local**:

- Các chức năng được tách thành module Python.
- Các module chạy trong cùng một process khi đánh giá.
- File hệ thống được dùng thay cho database.
- Chưa có API để tích hợp trực tiếp với web hoặc mobile.
