# Giải thích thay đổi và workflow HandSign

## Cập nhật: tính sẵn ngưỡng vùng khi build

`python build_calibration.py` hiện tạo calibration định dạng 3. Với mỗi label,
builder chọn các cặp mẫu như trước, suy ra vùng cần chấm và gọi
`calculate_region_thresholds()` để tính P90. Kết quả được lưu cùng label trong
`artifacts/label_calibration.json`: `required_regions`, `region_thresholds` và
`region_threshold_sample_count`. Dưới 3 cặp thì ngưỡng là null và label chưa đủ
dữ liệu để chấm vùng.

`SignEvaluator._region_thresholds()` chỉ đọc các giá trị đã lưu. Mỗi user mới
vẫn được so với mẫu chuẩn, nhưng không tính lại ngưỡng từ các cặp chuẩn.
Calibration cũ hoặc thiếu/sai ngưỡng yêu cầu chạy lại `python build_calibration.py`.
Ngưỡng P90, công thức điểm và cách tạo feedback được giữ nguyên. Các mô tả tính
ngưỡng lúc chấm ở phần lịch sử bên dưới không còn áp dụng.

Khi đổi dữ liệu chuẩn hoặc cách tính khoảng cách, cần cập nhật cache
`result2/info.npy` tương ứng rồi build lại; builder không tự tạo lại cache này.
Phân phối DTW tham khảo khi chỉ chấm một tay vẫn có thể được tính lúc đánh giá;
đây là phần riêng, không phải ngưỡng vùng dùng ra điểm.

## Cập nhật: điểm vùng có trọng số

Nhận diện khác target và confidence ≥ 70% hiện trả `incorrect_label`, score
null, feedback về nhãn và bỏ qua chấm vùng. Khác target dưới 70% vẫn chấm luyện
tập, cảnh báo chưa xác nhận đúng từ, không kết luận `correct`. Trùng target vẫn
dùng điều kiện confidence ≥ 50%, margin ≥ 12% nếu có margin. Đây là quy tắc mới
thay cho các mô tả mismatch vẫn trả điểm ở phần lịch sử bên dưới.

Tracking hiện chỉ cảnh báo theo tỷ lệ phát hiện tay, không chặn chấm ở bất kỳ
tỷ lệ nào (kể cả 0%) nếu chuỗi landmark vẫn hợp lệ. `low_quality` vẫn được giữ
để thể hiện chất lượng dữ liệu. Landmark thiếu có thể ảnh hưởng điểm. Dữ liệu
toàn số 0 vẫn không chấm; thiếu calibration vùng vẫn trả score null. Các mô tả
chặn theo tỷ lệ tracking bên dưới là hành vi lịch sử.

Phần mô tả percentile/validation làm điểm chính bên dưới là thiết kế lịch sử.
Hiện `form.score` lấy trung bình có trọng số từ chính distance/threshold của
feedback: các tay cần dùng chia nhau 80%, mặt 20%. Nếu chỉ có một tay, tay đó
nhận 80%; tay bị bỏ qua không tham gia chấm. Nếu không chấm mặt thì các tay
chia nhau 100%.

Với `r = distance / threshold`: r ≤ 0.3 được 100 điểm; 0.3 < r ≤ 1 dùng
`100 - 30 * (r - 0.3) / 0.7`; r > 1 dùng `70 * 2^(1-r)`.
Ngưỡng bằng 0: distance bằng 0 được 100, distance lớn hơn 0 được 0.
Điểm từng vùng làm tròn hai chữ số, nhân trọng số rồi cộng và làm tròn tổng.
`excellent` từ 85, `good` từ 70, `needs_practice` từ 30, thấp hơn là
`far_from_reference`. Feedback từng vùng vẫn báo vượt ngưỡng độc lập với tổng.

`form.region_scores` và `form.region_weights` giải thích điểm tổng;
`decision_source = weighted_region_thresholds`. Percentile và DTW chỉ còn là
thông tin tham khảo; validation cũ không quyết định điểm mới. Thiếu ngưỡng vùng
cần chấm thì score là null và form.status là not_scored. Ít mẫu vẫn được cảnh
báo qua calibration_quality và feedback.

Tài liệu này mô tả code đang có trong C:\HandSign, đối chiếu với các file đã đọc trước khi sửa trong cuộc hội thoại. Phần so sánh lịch sử dựa trên cả cuộc hội thoại: một số file đã là untracked và record.py đã có thay đổi của bạn trước khi mình làm. Vì vậy git diff với HEAD không đồng nghĩa toàn bộ chênh lệch là do mình tạo.

Mục đích là giải thích nguồn dữ liệu, vai trò từng file, từng phép tính và các giới hạn thực tế. Đây là bản giải thích; trong lượt này mình không thay thuật toán chấm điểm.

## 1. Hiện hệ thống làm gì?

Một lần đánh giá tạo ba nhóm thông tin:

| Nhóm | Câu hỏi được xử lý | Nguồn |
| --- | --- | --- |
| Tracking | Camera có phát hiện được tay/mặt trong các frame không? | Mask do MediaPipe tạo trong lúc quay |
| Recognition | Video có vẻ là từ nào? Có trùng từ được yêu cầu không? | VideoMAE có sẵn |
| Form và feedback | Chuỗi landmark gần các mẫu của từ mục tiêu đến đâu? Vùng nào vượt ngưỡng? | DTW, bộ mẫu và hai file calibration |

VideoMAE là mô hình học sâu. DTW là thuật toán so sánh chuỗi. Câu feedback hiện được ghép từ mẫu câu viết sẵn và các phép so sánh số học; chưa có LLM viết lời khuyên, chưa có mô hình riêng được huấn luyện để chẩn đoán lỗi ngón tay.

Không có quá trình fine-tune VideoMAE hoặc train mô hình neural network mới trong những thay đổi mình đã làm.

## 2. Những file đã thay đổi

### 2.1. File được tạo mới

| File | Vai trò và lý do có file này |
| --- | --- |
| [sign_eval/__init__.py](C:/HandSign/sign_eval/__init__.py) | Khai báo package và xuất SignEvaluator, EvaluationResult cho code khác import |
| [sign_eval/landmarks.py](C:/HandSign/sign_eval/landmarks.py) | Gom định dạng landmark, đọc NPY/NPZ, kiểm tra đầu vào, normalize, trim, DTW và lỗi từng vùng |
| [sign_eval/calibration.py](C:/HandSign/sign_eval/calibration.py) | Chọn reference hợp lệ, loại bản trùng khỏi danh sách sử dụng, tạo phân phối leave-one-out |
| [sign_eval/validation.py](C:/HandSign/sign_eval/validation.py) | Tìm ngưỡng từ các clip được người dùng gán đúng/sai |
| [sign_eval/evaluator.py](C:/HandSign/sign_eval/evaluator.py) | Đọc calibration, chấm chuỗi mới, tổng hợp recognition, form, feedback thành JSON |
| [recognizer.py](C:/HandSign/recognizer.py) | Gom logic VideoMAE từ Video-Text.py cũ thành hàm có thể gọi lại |
| [evaluate.py](C:/HandSign/evaluate.py) | Lệnh chính để nhận tham số, gọi nhận diện nếu có video, chấm và ghi JSON |
| [build_calibration.py](C:/HandSign/build_calibration.py) | File khởi chạy main() của sign_eval.calibration |
| [build_validation.py](C:/HandSign/build_validation.py) | File khởi chạy main() của sign_eval.validation |
| [tests/test_landmarks.py](C:/HandSign/tests/test_landmarks.py) | Năm kiểm tra đơn vị cho các phép xử lý nền |
| artifacts/label_calibration.json | Dữ liệu calibration đã build từ corpus và cache cũ |
| artifacts/user_validation_calibration.json | Ngưỡng đã fit từ 5 clip đúng và 5 clip sai của Cảm ơn |

Các file build ở ngoài ngắn vì thuật toán nằm trong package. Ví dụ:

~~~python
# build_validation.py
from sign_eval.validation import main

if __name__ == "__main__":
    main()
~~~

Lệnh gọi theo thứ tự:

~~~text
python build_validation.py ...
    -> sign_eval.validation.main()
    -> đọc tham số CLI
    -> build_validation_calibration(...)
    -> tính ngưỡng và lưu JSON
~~~

### 2.2. File có sẵn được sửa

| File | Trước khi mình sửa | Hiện tại |
| --- | --- | --- |
| [record.py](C:/HandSign/record.py) | Quay và chạy xử lý ngay khi import; lưu toàn sequence dù video đã trim; đường dẫn lưu hard-code vào eval_set/incorrect/005 | Có main guard; khởi tạo detector trong hàm; giữ mask; trim sequence cùng đoạn video; lưu vào data/user; giải phóng tài nguyên trong finally |
| [dtw.py](C:/HandSign/dtw.py) | Tự chứa scoring tuyến tính, ngưỡng vùng chung, các đường dẫn thử nghiệm cố định | Wrapper gọi SignEvaluator; nhận sequence và label_dir từ CLI |
| [baseline.py](C:/HandSign/baseline.py) | Tạo result2/label_distance_distribution.json từ info.npy; dtw.py lúc đó không dùng output này | Alias gọi cùng main() với build_calibration.py |
| [Video-Text.py](C:/HandSign/Video-Text.py) | Toàn bộ tải model, đọc video và in dự đoán nằm trực tiếp trong file | Wrapper gọi recognizer.main() |
| [requirements.txt](C:/HandSign/requirements.txt) | Danh sách pin cũ thiếu nhiều thư viện được import thực tế | Khai báo NumPy, FastDTW, MediaPipe, OpenCV, Torch, Transformers, Hugging Face Hub và Decord |
| [README.md](C:/HandSign/README.md) | Nội dung rất ít | Thêm cấu trúc, lệnh chạy và giải thích output |

Trong record.py, việc lấy 48 điểm 2D và dùng z-score đã có ở bản bạn đang làm trước khi giao sửa. Mình giữ lại biểu diễn này để dùng với dataset hiện có. Không nên đọc git diff từ bản commit cũ rồi hiểu rằng mình mới chuyển toàn bộ từ 3D sang 2D trong phiên này.

Các wrapper giữ được một số cách gọi ở mức tên file/hàm; không tương thích hoàn toàn với mọi API cũ. dtw.score_user() hiện trả score, danh sách reference rút gọn và list feedback có cấu trúc. Code khác phụ thuộc vào các hàm phụ hoặc dictionary feedback cũ cần được cập nhật.

### 2.3. File đã xóa và dữ liệu được giữ

- Xóa pipe.py: demo webcam cũ không được pipeline mới gọi. File này có trong Git nên có thể lấy lại nội dung từ lịch sử.
- Xóa thresholds.py: script tạo ngưỡng vùng chung, hard-code đường dẫn Kaggle. File này lúc đầu là untracked, nên Git không bảo đảm khôi phục được; nội dung đã xuất hiện trong cuộc hội thoại.
- Xóa hai JSON kiểm thử tạm trong artifacts và các thư mục __pycache__ trong đợt trước. Cache có thể xuất hiện lại khi chạy Python; đó là bình thường.
- Giữ data/, các video gốc, các model .task, notebook Value_threshold.ipynb, PIPIEN.pdf, result2/, result3/ và các archive đang có.
- Giữ label_baseline.json cũ và result2/label_distance_distribution.json cũ. Evaluator mới không đọc hai JSON này.
- Không thay đổi do mình thực hiện đối với desktop.ini hoặc việc landmarks.zip đã bị xóa trước đó.
- Chưa sắp xếp lại toàn bộ thư mục dữ liệu; phần được tổ chức lại chủ yếu là mã nguồn.
- Chưa sửa .gitignore để bỏ qua cache, model hoặc video. .gitignore hiện chỉ chứa mục OpenFace-3.0/.

## 3. Dữ liệu lấy từ đâu?

| Đầu vào / tài nguyên | Nguồn thực tế | Chỗ sử dụng |
| --- | --- | --- |
| target_label | Bạn truyền qua --target-label, ví dụ Cảm ơn | Chọn label cần học và folder reference |
| Video mới | Webcam số 0 qua OpenCV trong record.py | Lưu video và trích landmark |
| hand_landmarker.task | File model đã có tại root project | MediaPipe tìm tay và landmark |
| face_landmarker.task | File model đã có tại root project | MediaPipe tìm mặt và landmark |
| pose_landmarker_lite.task | Có sẵn trên đĩa nhưng pipeline hiện không load | Chưa dùng vai, hông hoặc pose toàn thân |
| data/landmarks (1)/<label>/*.npy | Corpus landmark đã có trong workspace | Reference của từng label |
| result2/info.npy | Cache cặp reference-reference đã có trước khi mình sửa | Xây phân phối chuẩn và chọn cặp để tính ngưỡng vùng |
| data/eval_set/correct/*.npy | Các clip đã được đặt vào nhóm đúng | Fit ngưỡng thực hành |
| data/eval_set/incorrect/*.npy | Các clip đã được đặt vào nhóm sai | Fit ngưỡng thực hành |
| Model VideoMAE | Repo được cấu hình trong code: star092304/vi-sign-language-videomae-base | Nhận diện nhãn từ video |
| classifier_sequential.pth | File checkpoint tải từ cùng repo VideoMAE | Nạp trọng số của kiến trúc classifier tương ứng |

Mình không tạo dataset reference gốc và không tự gán lại đúng/sai cho các clip của bạn. Code tin vào tên folder và label do bạn cung cấp.

Trong workspace hiện chưa thấy script hoàn chỉnh đã tạo corpus landmark gốc hoặc đã sinh result2/info.npy. Vì vậy chưa xác minh được toàn bộ nguồn dataset, người biểu diễn, quy trình augmentation, cách mirror hoặc cách resample lúc tạo reference. Không thể khẳng định tất cả reference được xử lý giống webcam chỉ vì cùng shape.

Cache info.npy hiện có shape (46651, 4). Mỗi hàng là:

~~~text
[label, file_a, file_b, distance]
~~~

Distance ở đây là khoảng cách đã được tính và lưu trước đó. Builder mới đọc số này; nó không tự tính lại toàn bộ DTW cho corpus.

Mình đối chiếu ba cặp của Cảm ơn bằng hàm DTW hiện tại và thấy các giá trị khớp trong sai số số thực. Đây là kiểm tra một phần, chưa xác minh mọi cặp trong cache.

## 4. Luồng thực thi thật

Có hai giai đoạn:

~~~text
CHUẨN BỊ
reference NPY + info.npy
    -> build_calibration.py
    -> artifacts/label_calibration.json

clip đúng/sai + calibration reference
    -> build_validation.py
    -> artifacts/user_validation_calibration.json

SỬ DỤNG
webcam -> record.py -> original_video.mp4 + trimmed_video.mp4
                       sequence.npy + record.npz

evaluate.py:
    đọc CLI / build calibration nếu chưa có
    -> nếu có --video: recognizer.predict_video(video)
    -> SignEvaluator.evaluate(sequence, target_label, recognition)
    -> kiểm tra tracking
    -> nếu tracking đủ: DTW + score + form + feedback
    -> tổng hợp status
    -> ghi evaluation.json
~~~

Đây là thứ tự chạy thực tế của [evaluate.py](C:/HandSign/evaluate.py:21). Nhận diện video đang chạy trước khi evaluator kiểm tra tracking.

Evaluator vẫn tính form và feedback khi recognition là mismatch hoặc uncertain, miễn tracking qua kiểm tra. Lời giải thích trước rằng hệ thống chỉ chấm form sau khi nhận diện đúng nhãn mô tả hướng thiết kế mong muốn, chưa đúng với code đang chạy.

Nếu không truyền --video, recognition là not_run và trạng thái cuối là needs_label_confirmation, dù vẫn có thể tính form.

Nếu có --video nhưng model hoặc dependency lỗi, CLI hiện sẽ báo lỗi và dừng. Chưa có try/except tự động chuyển sang chỉ chấm form.

## 5. Video webcam được xử lý thế nào?

Nguồn: [record.py](C:/HandSign/record.py:137).

### 5.1. Bắt đầu ghi

1. Mở camera số 0 bằng cv2.VideoCapture(0).
2. Tạo VideoWriter MP4 với codec mp4v, FPS khai báo cố định là 30.
3. Bấm S để đếm ngược 3 giây.
4. Sau đếm ngược, mỗi frame đọc được được lật ngang bằng cv2.flip(frame, 1).
5. Ghi frame vào video gốc.
6. Chuyển màu BGR sang RGB để đưa vào MediaPipe.
7. Lấy landmark tay/mặt và mask.
8. Vẽ landmark lên ảnh hiển thị.
9. Bấm Q để kết thúc.

writer.write(frame) nằm trước bước vẽ landmark và dòng chữ RECORDING. Vì vậy video lưu là ảnh webcam đã mirror, không có bộ xương overlay của giao diện.

MediaPipe được cấu hình ở IMAGE mode và gọi detect() theo từng frame. Chưa dùng VIDEO mode với timestamp để theo dõi chuỗi ở API này. Cả detection confidence và presence confidence đang đặt 0.5. Những số này là cấu hình detector, không phải điểm động tác.

### 5.2. Lấy 48 landmark

Mỗi frame có shape (48, 2):

| Khoảng index trong sequence | Số điểm | Nội dung |
| --- | --- | --- |
| 0..20 | 21 | Tay có handedness là Left |
| 21..41 | 21 | Tay có handedness còn lại, thường là Right |
| 42..47 | 6 | Các điểm mặt được chọn |

Code lưu landmark.x và landmark.y, không lưu landmark.z hoặc hand world coordinates. Với tọa độ ảnh chuẩn hóa của detector, x gắn với chiều ngang và y với chiều dọc ảnh.

Ví dụ 54 frame có:

~~~text
landmarks.shape = (54, 48, 2)
54 * 48 * 2 = 5184 giá trị tọa độ
~~~

Sáu index mặt là 33, 263, 1, 13, 234, 454. Code đặt tên tương ứng là mắt phải, mắt trái, chóp mũi, điểm miệng và hai bên mặt. Hai tên biến EAR không có nghĩa đang đo tai bằng một ear tracker riêng.

Bộ sáu điểm này là thông tin vị trí/hình học rất thưa. Không đủ căn cứ để chẩn đoán đầy đủ biểu cảm như nhíu mày, độ mở miệng hay sắc thái ngữ pháp của ngôn ngữ ký hiệu.

### 5.3. Mask tracking

Mỗi frame có ba boolean:

~~~python
validity[i] = [left_detected, right_detected, face_detected]
~~~

Shape mask là (T, 3). Không phải mỗi landmark có một confidence riêng.

Khi không phát hiện một vùng, vùng đó vẫn được điền tọa độ 0 trước khi normalize; mask của vùng là False.

Handedness lấy trực tiếp từ detector. Chưa có thuật toán giữ danh tính tay liên tục giữa các frame. Trường hợp hai detection cùng rơi vào một slot, code có thể ghi đè slot đó.

### 5.4. Trim

Nguồn: [trim_to_active_hands](C:/HandSign/sign_eval/landmarks.py:88).

Tạo active[i] = left_detected[i] OR right_detected[i]. Lấy frame đầu tiên và cuối cùng có active=True:

~~~text
sequence_trim = sequence[start:end + 1]
mask_trim = mask[start:end + 1]
video_trim = các frame start..end
~~~

Mình sửa điểm lệch cũ: trước đây video được trim nhưng NPY vẫn lưu cả sequence chưa trim. Hiện video và landmark được cắt cùng khoảng frame; normalize được thực hiện sau khi trim.

Đây chỉ là trim theo sự xuất hiện của tay. Chưa phát hiện chính xác lúc bắt đầu/kết thúc ký hiệu. Nếu tay vẫn hiện nhưng bạn đứng chờ, đoạn chờ vẫn có thể ở lại. Frame mất tay ở giữa đoạn vẫn được giữ.

Video hiện có trên máy lúc đối chiếu:

| File | Số frame | Kích thước | FPS lưu |
| --- | --- | --- | --- |
| original_video.mp4 | 106 | 640 x 480 | 30 |
| trimmed_video.mp4 | 54 | 640 x 480 | 30 |

54/30 = 1.8 giây phát lại. Không thể dùng đó để khẳng định bạn thực hiện ngoài đời đúng 1.8 giây, vì code không lưu timestamp thật và mỗi lượt xử lý camera có thể chậm hơn 1/30 giây.

### 5.5. Normalize

Nguồn: [zscore_normalize_sequence](C:/HandSign/sign_eval/landmarks.py:78).

Tính riêng cho trục x và y nhưng gộp tất cả frame và tất cả 48 điểm:

~~~text
mu_x = mean của T * 48 giá trị x
sigma_x = std của T * 48 giá trị x, NumPy mặc định ddof=0

x_norm = (x - mu_x) / max(sigma_x, 1e-8)

mu_y, sigma_y và y_norm tính tương tự
~~~

Đây là z-score toàn clip. Sau phép này, tọa độ có thể âm hoặc lớn hơn 1.

Điểm chưa được sửa đầy đủ: các tọa độ 0 do mất detection vẫn tham gia tính mean/std. Mask đã được lưu nhưng chưa dùng để loại missing values khỏi normalize, DTW hoặc lỗi từng vùng. Hai lời mô tả này phải được phân biệt rõ: có lưu mask chưa có nghĩa phép đo đã xử lý missing landmarks đúng.

Chưa có normalize theo tâm hai mắt, khoảng cách hai mắt, vai hoặc thân người. Chưa nội suy gap ngắn, chưa smoothing, chưa resample landmark về một độ dài chung, chưa tách độ cong ngón tay hoặc vận tốc.

### 5.6. Lưu đầu ra capture

| File | Nội dung |
| --- | --- |
| data/user/video_user/original_video.mp4 | Các frame đã mirror trong thời gian recording, trước trim |
| data/user/video_user/trimmed_video.mp4 | Video cùng đoạn active với sequence |
| data/user/sequence_user/sequence.npy | Landmark đã trim và normalize |
| data/user/sequence_user/record.npz | Hai array landmarks và validity của cùng đoạn |

NPZ hiện không chứa raw landmarks, timestamps, FPS, start/end ở video gốc, label mục tiêu, tên camera hoặc version normalize. NPY đơn lẻ không có mask.

Tên file đầu ra đang cố định. Một lần quay mới có thể ghi đè bản user trước đó; chưa có thư mục attempt ID riêng.

## 6. Nhận diện VideoMAE lấy gì và tính gì?

Nguồn: [recognizer.py](C:/HandSign/recognizer.py).

### 6.1. Nguồn model và cấu hình

Code lấy processor/config/model từ star092304/vi-sign-language-videomae-base, rồi lấy classifier_sequential.pth từ cùng repo. Logic này được chuyển từ Video-Text.py cũ, không phải model do mình train.

Cache model trên máy có snapshot 9759fc78b5d69625317bb658b4104040f9427fa3. Mình đọc config.json và preprocessor_config.json ở snapshot đó để kiểm tra các số dưới đây.

Model config đang có 100 nhãn trong id2label, 16 frame đầu vào, image_size=224, hidden_size=768 và 12 encoder layer. Nhãn dự đoán được tra từ model.config.id2label, không lấy từ thứ tự folder reference.

Code chưa pin revision của Hugging Face repo khi tải. Snapshot trên máy là nguồn cấu hình kiểm tra trong lượt này, không phải bảo đảm mọi lần cài mới đều lấy cùng phiên bản.

### 6.2. Chọn 16 frame video

Video được đọc bằng Decord trên CPU:

~~~python
indices = np.linspace(0, total_frames - 1, 16).astype(int)
~~~

Với video 54 frame, index là:

~~~text
0, 3, 7, 10, 14, 17, 21, 24, 28, 31, 35, 38, 42, 45, 49, 53
~~~

Đây là 16 frame rải đều trên video. DTW vẫn dùng toàn bộ 54 frame landmark; hai nhánh dùng biểu diễn đầu vào khác nhau. Video rất ngắn có thể được chọn lặp frame do chuyển index về integer.

### 6.3. Tiền xử lý ảnh

Cấu hình processor được cache quy định:

1. Resize cạnh ngắn nhất về 224.
2. Center crop ảnh 224 x 224.
3. Rescale giá trị pixel bằng 1/255.
4. Normalize từng kênh RGB với mean = [0.485, 0.456, 0.406] và std = [0.229, 0.224, 0.225].

Công thức từng kênh:

~~~text
pixel_model[c] = (pixel_rgb[c] / 255 - mean[c]) / std[c]
~~~

Đây là normalize ảnh cho VideoMAE, khác hoàn toàn z-score landmark. Tensor theo cách dùng này có bố cục một video, 16 frame, 3 kênh, 224 x 224.

Center crop có thể cắt mất tay ở rìa. Code đang dùng processor của model; chưa thêm kiểm tra tay còn nằm đủ trong crop.

### 6.4. Nạp trọng số và dự đoán

Classifier được thay bằng:

~~~text
LayerNorm(768)
    -> Dropout(0.3)
    -> Linear(768, số nhãn)
~~~

Code đổi tên q_bias thành query.bias, v_bias thành value.bias, thêm key.bias bằng 0, rồi gọi load_state_dict(strict=False). Đây là thao tác tương thích định dạng checkpoint kế thừa từ code cũ.

Chưa kiểm tra và bắt lỗi missing/unexpected keys sau bước nạp trọng số. Dùng strict=False không đủ để chứng minh mọi lớp đều đã được nạp đúng. Chỗ này còn cần kiểm chứng khi đổi version thư viện hoặc checkpoint.

model.eval() đưa model vào chế độ suy luận; torch.no_grad() tắt tính gradient. Code hiện không tự chuyển model/tensor sang CUDA.

Model xuất logits z. Softmax được tính:

~~~text
p[c] = exp(z[c]) / sum_j(exp(z[j]))
~~~

Lấy top-5 rồi trả:

~~~text
predicted_label = tên nhãn đứng đầu
confidence = p_top1
margin = p_top1 - p_top2
~~~

Với ví dụ bạn gửi: confidence=0.8143, margin=0.7557. Suy ra p_top2 xấp xỉ 0.0586 dựa trên số đã làm tròn.

0.8143 là softmax score. Chưa có bước hiệu chỉnh xác suất trên bộ kiểm chứng độc lập, nên không thể suy ra chắc chắn rằng dự đoán kiểu này đúng 81.43% ngoài thực tế. Docstring trong recognizer nói calibrated là diễn đạt quá mạnh.

### 6.5. Quyết định match

Nguồn: [SignEvaluator._recognition_status](C:/HandSign/sign_eval/evaluator.py:166).

| Điều kiện | recognition.status |
| --- | --- |
| Không truyền recognition | not_run |
| Thiếu predicted_label | uncertain |
| confidence được truyền và nhỏ hơn 0.50 | uncertain |
| margin được truyền và nhỏ hơn 0.12 | uncertain |
| Qua các điều kiện và nhãn bằng target | match |
| Qua các điều kiện và nhãn khác target | mismatch |

0.50 và 0.12 là ngưỡng heuristic mình đặt trong code, chưa được học từ bộ validation.

So nhãn có chuẩn hóa Unicode NFC, chữ hoa/thường và khoảng trắng. Không bỏ dấu tiếng Việt hoặc tự ánh xạ đồng nghĩa.

Nếu gọi API trực tiếp mà chỉ truyền predicted_label, thiếu confidence/margin hiện chưa bị coi là không hợp lệ; các kiểm tra tương ứng bị bỏ qua. CLI recognizer bình thường trả cả hai số.

## 7. Kiểm tra tracking trước khi chấm

Nguồn: [tracking_report](C:/HandSign/sign_eval/landmarks.py:103).

Reader kiểm tra shape (T,48,2), T >= 2 và giá trị hữu hạn. Shape sai, chuỗi quá ngắn, NaN hoặc Inf gây ValueError; CLI chưa chuyển tất cả những lỗi này thành JSON not_scored.

Với chuỗi hợp lệ về cấu trúc:

- Nếu tất cả landmark đều bằng 0: tracking=invalid, evaluator không chấm.
- Nếu là NPY không mask: tracking=unknown; evaluator vẫn chấm.
- Nếu có mask: tính tỷ lệ phát hiện từng vùng và tỷ lệ có ít nhất một tay.
- Có tay >= 70% frame: tracking=ok.
- Có tay < 70% frame: tracking=low_quality; evaluator không chấm.

70% là ngưỡng cấu hình do mình đặt, chưa tối ưu trên dữ liệu có nhãn chất lượng tracking.

Với đúng record.npz của bạn:

| Trường | Phép tính | Giá trị |
| --- | --- | --- |
| frame_count | Tổng frame sau trim | 54 |
| left_hand_detected_fraction | 38/54 | 0.7037 |
| right_hand_detected_fraction | 10/54 | 0.1852 |
| hand_detected_fraction | 48/54 | 0.8889 |
| face_detected_fraction | 54/54 | 1.0 |

0.8889 >= 0.70 nên tracking là ok. Nó không nói cả hai tay đều được thấy tốt.

Trong clip này, tổng 38+10 bằng đúng số 48 frame có ít nhất một tay. Không có frame nào được mask đánh dấu cả hai tay cùng hiện. Điều đó chưa đủ để kết luận lỗi kỹ thuật của bạn; có thể liên quan động tác một tay, handedness hoặc tracking. Cần xem video và yêu cầu của ký hiệu trước khi chẩn đoán.

Code hiện không biết mỗi từ cần một hay hai tay. Vì vậy không nên khuyên phải đưa tay phải vào khung chỉ dựa trên tỷ lệ thấp, cũng không nên hiểu feedback tay phải ok là tay phải chắc chắn làm đúng.

## 8. Calibration reference được tạo ra thế nào?

Nguồn: [sign_eval/calibration.py](C:/HandSign/sign_eval/calibration.py:77).

### 8.1. Chọn reference hợp lệ

Duyệt từng folder nhãn, rồi từng NPY theo thứ tự tên:

1. Đọc và ép kiểu float32.
2. Kiểm tra shape, số frame, NaN/Inf.
3. Loại sequence toàn số 0 khỏi danh sách sử dụng.
4. Tính SHA-256 của array.tobytes().
5. Nếu đã gặp cùng hash trong chính label đó, loại bản sau khỏi danh sách.
6. Giữ tên file đầu tiên của mỗi nhóm.

Đây là so bytes của dữ liệu array sau khi đọc/ép float32, không nhất thiết là toàn bộ bytes file NPY gồm header. Không có so độ tương đồng video và không dedupe xuyên các label.

Các file không bị xóa khỏi dataset. Đường dẫn bị loại và lý do được lưu trong exclusions của calibration.

Artifact hiện ghi nhận:

| Mục | Số lượng |
| --- | --- |
| Nhãn | 100 |
| Reference ban đầu trong corpus được thống kê | 2.660 |
| Trùng array bị loại khỏi danh sách | 610 |
| Sequence toàn số 0 bị loại | 3 |
| Reference dùng sau lọc | 2.047 |
| Cặp trong info.npy trước lọc | 46.651 |
| Cặp dùng sau lọc | 27.081 |
| Label dưới 10 reference sau lọc | 27 |

Số cặp cache bằng 0 trước lọc là 869. Một nhóm gồm nhiều bản trùng tạo ra nhiều cặp 0, nên 869 cặp không đồng nghĩa 869 file trùng.

### 8.2. Phân phối leave-one-out top-k

Chỉ giữ hàng cache mà cả file_a và file_b đều còn trong danh sách reference.

Với mỗi reference R_i của label:

~~~text
1. Lấy khoảng cách từ R_i tới các reference khác trong cache.
2. Sắp xếp tăng dần.
3. Lấy k khoảng cách nhỏ nhất, mặc định k=3.
4. b_i = trung bình các khoảng cách được chọn.
~~~

Reference i được xem như một user giả, so với các mẫu khác. Không lấy self-distance nếu cache đúng là các cặp reference khác nhau.

~~~text
B_label = sort([b_1, b_2, ..., b_n])
~~~

B_label được lưu thành loo_top_k_distances.

Ví dụ Cảm ơn:

- Ban đầu có 43 reference.
- Sau lọc còn 33.
- Có 33 * 32 / 2 = 528 cặp khác nhau.
- Tạo 33 giá trị b_i.
- reference_count=33 và calibration_count=33.

Đây không phải 33 giá trị confidence của model và không phải 33 lần quay của bạn.

Nếu label có ít mẫu, builder lấy min(top_k, số neighbor có sẵn). Vì vậy label có 3 reference chỉ có 2 neighbor cho mỗi điểm chuẩn, trong khi user mới có thể được so trung bình 3 reference. Hai thống kê không hoàn toàn tương đương ở nhóm ít mẫu.

### 8.3. calibration_quality

~~~text
reference_count >= 10 -> ok
reference_count < 10  -> low_sample
~~~

Chữ ok chỉ xuất phát từ kiểm tra số lượng mẫu. Chưa đo chất lượng ground truth, đa dạng người, framing, độ cân bằng hay độ chính xác thực tế.

### 8.4. Những gì builder chưa kiểm tra

- Không tự tính cặp mới khi thêm reference.
- Không kiểm chứng cache còn đúng sau khi sửa nội dung file nhưng giữ nguyên tên.
- Không kiểm chứng cache đủ toàn bộ cặp, không trùng hàng hoặc không chứa self-pair.
- Không gắn hash dataset/phiên bản thuật toán vào artifact để evaluator phát hiện dữ liệu cũ.
- Tách bản sao chính xác chưa loại được các clip rất gần nhau cùng người hoặc cùng video gốc.

Vì vậy hướng dẫn cũ chỉ nói chạy lại builder sau khi thêm mẫu là thiếu. Khi thêm hoặc đổi mẫu, cần cập nhật cache DTW phù hợp trước rồi build lại calibration; hiện chưa có CLI hoàn chỉnh để tự tái tạo cache này.

## 9. DTW và comparison_distance

Nguồn: [calculate_dtw](C:/HandSign/sign_eval/landmarks.py:147) và [evaluate](C:/HandSign/sign_eval/evaluator.py:328).

### 9.1. Khoảng cách hai frame

Flatten mỗi frame từ (48,2) thành 96 số.

~~~text
frame_distance(u_i, r_j)
  = sqrt(sum từ k=1..96 của (u_i[k] - r_j[k])^2)
~~~

Code dùng np.linalg.norm(a-b). Mọi tọa độ có trọng số bằng nhau.

Vì có 42 điểm tay và chỉ 6 điểm mặt, không thể nói ba vùng được cân bằng trọng số ngang nhau. Không có weight học riêng cho từng label.

### 9.2. Ghép theo thời gian

FastDTW nhận hai chuỗi frame và trả một đường ghép P gồm các cặp (i,j). Ví dụ:

~~~text
P = [(0,0), (1,0), (2,1), (3,2), ...]
~~~

Nhiều frame user có thể ghép với cùng một frame mẫu hoặc ngược lại. Điều này cho phép so các chuỗi dài khác nhau.

Hệ thống dùng thư viện FastDTW, không tự cài thuật toán DP chính xác. Code không truyền ràng buộc temporal window riêng hoặc penalty cho tốc độ.

Khoảng cách một cặp clip:

~~~text
d(user, reference)
  = sum(frame_distance(u_i, r_j) với (i,j) trong P) / len(P)
~~~

Đây là khoảng cách trên tọa độ đã normalize, không có đơn vị cm hoặc độ. Chia cho độ dài path giúp giảm ảnh hưởng độ dài tổng nhưng không chứng minh mọi khác biệt tốc độ đều được xử lý đúng.

### 9.3. Chọn top-3 reference

So user với mọi reference trong label mục tiêu, sắp xếp khoảng cách từ nhỏ đến lớn:

~~~text
D_user = mean(k khoảng cách nhỏ nhất)
k = min(top_k, số reference)
~~~

D_user chính là comparison_distance.

Ví dụ của bạn:

| Mẫu tham chiếu | DTW |
| --- | --- |
| 679364.npy | 6.038819 |
| 285652.npy | 6.791675 |
| 950155.npy | 7.011356 |

~~~text
D_user = (6.038819 + 6.791675 + 7.011356) / 3
       = 6.61395
~~~

Các giá trị JSON đã làm tròn; nội bộ việc tính top-k dùng số chưa làm tròn.

Chỉ so với mẫu Cảm ơn vì target_label là Cảm ơn. DTW này không tự tìm từ tốt nhất trong cả 100 label; đó là nhiệm vụ nhánh nhận diện video.

Đã bỏ đường dẫn demo mặc định lấy user từ chính folder reference, nhưng chưa thêm cơ chế loại self-match tổng quát. Lượt kiểm tra này dùng 141832.npy làm user thì chính 141832.npy vẫn xuất hiện với distance=0.0. Điều này có thể làm đẹp điểm khi dùng reference để test.

## 10. Score được tính chính xác ra sao?

Nguồn: [SignEvaluator._validation_score](C:/HandSign/sign_eval/evaluator.py:141).

### 10.1. Công thức trước khi sửa

dtw.py cũ dùng:

~~~text
baseline = (median(pair_distances) + P75(pair_distances)) / 2
score_old = 100 * clip(1 - D_user / baseline, 0, 1)
~~~

Nếu thiếu baseline hoặc baseline <=0, hàm cũ trả 100. Điểm đó không biểu thị dữ liệu đủ để đánh giá.

Công thức tuyến tính cũng coi khoảng cách bằng mức baseline nội bộ là 0 điểm, dù baseline chỉ phản ánh khác biệt giữa các mẫu cùng nhãn.

baseline.py cũ đã tạo một JSON phân phối khác, nhưng dtw.py cũ không dùng JSON đó. Mình đã thay luồng này bằng một calibration được evaluator thật sự đọc.

### 10.2. Công thức hiện tại

Khi label có bộ validation đúng/sai, form.score được quy đổi trực tiếp từ khoảng cách DTW. Đặt:

- D: trung bình khoảng cách DTW tới top-3 mẫu gần nhất (D chưa làm tròn).
- a: correct_median_distance, trung vị khoảng cách của nhóm đúng.
- b: threshold_distance, ngưỡng đạt.
- c: incorrect_median_distance, trung vị khoảng cách của nhóm sai.

Với 0 < a < b < c:

~~~text
                 / 100 - 20 * D/a                     nếu 0 <= D <= a
score(D) =       | 80 - 20 * (D-a)/(b-a)                nếu a < D <= b
                 | 60 - 30 * (D-b)/(c-b)                nếu b < D <= c
                 \ 30 * 2 ** (-(D-c)/(c-b))             nếu D > c
~~~

Các mốc D = 0, a, b, c tương ứng 100, 80, 60, 30 điểm. Ba đoạn đầu nội suy tuyến tính; đoạn cuối giảm dần về 0. Khoảng cách tăng thì điểm không tăng. Code làm tròn điểm tới hai chữ số thập phân, rồi dùng chính điểm này để xếp loại, nên điểm hiển thị và form.status luôn nhất quán trong nhánh validation. Làm tròn có thể đưa giá trị cực sát ngưỡng lên đúng 60 hoặc 80 điểm.

80/60/30 là các mốc quy đổi được chọn cho thang điểm luyện tập, không phải xác suất làm đúng hay phần trăm kỹ thuật đúng. Code yêu cầu các mốc hữu hạn và đúng thứ tự 0 < a < b < c; nếu không thỏa thì báo lỗi để hiệu chỉnh lại validation, không tự gán điểm. Bộ Cảm ơn hiện mới có 5 clip đúng và 5 clip sai nên độ tin cậy vẫn bị giới hạn bởi số mẫu.

Điểm percentile cũ được giữ riêng cho phân tích:

~~~text
form.reference_percentile = 100 * count(r_i >= D) / số giá trị r_i
~~~

r_i là các mốc leave-one-out của label, với 33 mốc cho Cảm ơn. Đây là thứ hạng tương đối trong phân phối tham chiếu; các giá trị bằng D vẫn được tính.

Label chưa có validation vẫn dùng percentile làm form.score và giữ quy tắc xếp loại cũ, vì chưa có a, b, c để áp dụng công thức mới. form.decision_source cho biết nhánh đang sử dụng. Các ví dụ score ở những phần lịch sử khác của tài liệu là kết quả trước thay đổi này; với label có validation, hãy đọc chúng là reference_percentile.

### 10.3. Ví dụ Cảm ơn: từ 15.15 thành 81.96 điểm

~~~text
a = 4.627085
b = 4.940030
c = 6.332769
D = 4.172959 < a

form.score = 100 - 20 * (4.172959 / 4.627085) = 81.96
form.status = excellent
form.reference_percentile = 100 * 5/33 = 15.15
~~~

Kết quả cũ D = 6.61395 từng có score = 3.03 vì chỉ 1/33 mốc tham chiếu lớn hơn hoặc bằng D. Theo công thức mới, D > c nên điểm khoảng 26.08, xếp needs_practice; 3.03 được giữ trong reference_percentile.

Điểm chính không còn nhảy theo nấc 100/33 khi có validation. Việc đổi thang điểm không sửa lỗi đầu vào như flip khác quy ước dataset hoặc video và sequence không cùng lần quay.

## 11. Ngưỡng validation 4.94003 lấy ở đâu?

Nguồn: [sign_eval/validation.py](C:/HandSign/sign_eval/validation.py:41).

### 11.1. Dữ liệu sử dụng

Builder đọc file NPY trực tiếp trong hai thư mục bạn truyền. Nó chưa đọc NPZ và chưa duyệt đệ quy thư mục con.

Mỗi file được evaluator tính D_user. Với Cảm ơn, số đọc lại hiện tại là:

| Ground truth theo folder | File | D_user |
| --- | --- | --- |
| correct | 1.npy | 4.627085 |
| correct | 2.npy | 5.136398 |
| correct | 4.npy | 4.778668 |
| correct | 5.npy | 4.037323 |
| correct | sequence.npy | 3.721604 |
| incorrect | 1.npy | 5.101393 |
| incorrect | 2.npy | 7.247324 |
| incorrect | 3.npy | 6.181418 |
| incorrect | 4.npy | 7.173324 |
| incorrect | 5.npy | 6.332769 |

Các folder được coi là ground truth đã gán bằng tay. Không có bước người đánh giá lại video trong builder. File data/eval_set/eval_results.json cũ không được dùng để fit ngưỡng; builder tính lại từ NPY.

### 11.2. Tìm ngưỡng

Gộp các D, lấy giá trị duy nhất, sắp tăng dần. Tạo ứng viên từ trung điểm các giá trị kề nhau và hai mốc ngoài biên.

Với mỗi ngưỡng t:

~~~text
D <= t -> dự đoán form đạt
D > t  -> dự đoán form chưa đạt

TPR = số clip correct có D <= t / tổng clip correct
TNR = số clip incorrect có D > t / tổng clip incorrect

balanced_accuracy = (TPR + TNR) / 2
~~~

Chọn ứng viên có balanced_accuracy lớn nhất. Nếu hòa, code giữ ứng viên gặp trước theo thứ tự duyệt.

Ngưỡng chọn cho Cảm ơn:

~~~text
t = (4.778668 + 5.101393) / 2
  = 4.9400305
lưu làm tròn 6 chữ số -> 4.94003
~~~

Ngưỡng này cho 4/5 clip correct được nhận và 5/5 clip incorrect bị loại:

~~~text
balanced_accuracy = (4/5 + 5/5) / 2 = 0.9
~~~

0.9 là kết quả trên chính bộ dùng để tìm t. Chưa có holdout theo người hoặc clip độc lập, nên không được công bố như accuracy 90% trên người dùng mới.

### 11.3. Những trường còn lại

~~~text
correct_count = 5
incorrect_count = 5
correct_median_distance = median(5 D correct) = 4.627085
incorrect_median_distance = median(5 D incorrect) = 6.332769
~~~

quality=ok khi cả hai nhóm có ít nhất 20 clip; nếu ít hơn thì low_sample. Đây là quy tắc số lượng do mình đặt, không phải đảm bảo thống kê.

Ngưỡng, hai median và quality được lưu vào artifacts/user_validation_calibration.json theo label. Đây không phải train lại VideoMAE.

Builder giữ các label khác đã có trong JSON và cập nhật mục label đang build. Khi đổi reference hoặc thuật toán khoảng cách, các ngưỡng validation cũ cũng cần được tính lại.

## 12. form.status có liên quan score như thế nào?

Nguồn: SignEvaluator._form_status trong [sign_eval/evaluator.py](C:/HandSign/sign_eval/evaluator.py).

Nếu có validation cho label, code tính điểm theo mục 10 rồi xếp loại từ chính form.score đã làm tròn:

| Điều kiện | form.status |
| --- | --- |
| score >= 80 | excellent |
| 60 <= score < 80 | good |
| score < 60 | needs_practice |

Với Cảm ơn hiện tại:

~~~text
D = 4.172959 -> score = 81.96 -> excellent
D = 4.627085 -> score = 80.00 -> excellent
D = 4.940030 -> score = 60.00 -> good
D = 6.332769 -> score = 30.00 -> needs_practice
~~~

Xếp loại theo điểm hiển thị tránh trường hợp người dùng thấy điểm thấp nhưng hệ thống lại báo excellent. Đây là xếp loại động tác; status tổng vẫn còn phụ thuộc nhận diện đúng nhãn.

Nếu chưa có validation:

| Điều kiện | form.status |
| --- | --- |
| Label reference ít hơn 10 mẫu | estimated_low_sample |
| Nếu đủ mẫu và score >= 75 | excellent |
| 45 <= score < 75 | good |
| 20 <= score < 45 | needs_practice |
| score < 20 | far_from_reference |

Các mốc 75/45/20 do mình chọn; chưa tối ưu từ dữ liệu ground truth.

Khi có validation, form.score là điểm luyện tập theo hàm chia đoạn và form.reference_percentile lưu percentile cũ. Khi chưa có validation, hai trường cùng là percentile. Không có phép trung bình hay nhân confidence nhận diện vào score.

Ví dụ lịch sử correct/1.npy từng có score=12.12 nhưng form.status=good thuộc thiết kế cũ. Sau thay đổi, 12.12 được giữ ở reference_percentile, còn điểm chính dùng cùng thang với xếp loại validation.

calibration_quality=ok và validation.quality=low_sample có thể cùng xuất hiện: một cái đếm 33 reference, cái kia đếm 5 correct + 5 incorrect. Code vẫn dùng ngưỡng low_sample để quyết định, chưa hạ kết luận xuống mức uncertain.

Ngoài ra nhánh excellent đang kiểm tra median trước threshold. Nếu một bộ validation tương lai tạo threshold thấp hơn correct_median, có thể có D vượt threshold nhưng vẫn excellent; code chưa kiểm tra thứ tự hai mốc này.

## 13. Feedback tay/mặt tính thế nào?

Nguồn: [region_distances_on_path](C:/HandSign/sign_eval/landmarks.py:158), [SignEvaluator._region_thresholds](C:/HandSign/sign_eval/evaluator.py:108) và [SignEvaluator._build_feedback](C:/HandSign/sign_eval/evaluator.py:211).

### 13.1. Khoảng cách vùng

Dùng đường ghép DTW toàn thân đã tính; không chạy một đường DTW độc lập cho mỗi tay.

Với một cặp frame (i,j) trên path và vùng g có m điểm:

~~~text
e_g(i,j) = mean theo p trong vùng g của
           sqrt((user[i,p,x]-ref[j,p,x])^2
              + (user[i,p,y]-ref[j,p,y])^2)
~~~

Nếu một frame user i ghép với nhiều frame mẫu j, lấy trung bình các e_g(i,j) để được trace_g[i]. Sau đó lấy trung bình trace trên các frame user.

Khi xét ba reference gần nhất:

~~~text
region_distance = median(mean_error_ref1,
                         mean_error_ref2,
                         mean_error_ref3)
~~~

Top-3 feedback hiện được viết cố định là 3, trong khi scoring top_k có thể cấu hình trong calibration.

Khoảng cách vùng là trung bình độ lệch từng điểm. Khoảng cách DTW toàn clip dùng norm của cả vector 96 số rồi trung bình path. Hai công thức khác nhau, nên không cộng ba distance vùng để ra comparison_distance.

### 13.2. Threshold vùng từ 24 cặp

Ở bước build calibration:

1. Xếp cặp reference theo DTW tổng tăng dần.
2. Giữ ceil(75% số cặp) có DTW nhỏ hơn.
3. Chọn tối đa 24 cặp rải đều theo index trong danh sách đó.
4. Lưu tên hai file của từng cặp vào region_pair_samples.

Với Cảm ơn: 528 cặp -> lấy 396 cặp ở đầu -> chọn 24 cặp.

Khi evaluator cần feedback lần đầu cho label, nó load lại 24 cặp, tính DTW và lỗi từng vùng. Sau đó:

~~~text
threshold_left  = quantile(24 lỗi tay trái, 0.90)
threshold_right = quantile(24 lỗi tay phải, 0.90)
threshold_face  = quantile(24 lỗi mặt, 0.90)
~~~

Kết quả được cache trong RAM của instance evaluator, chưa ghi threshold vùng thành artifact độc lập. Mỗi lần chạy CLI mới có thể phải tính lại.

24, 75% và quantile 90% là lựa chọn heuristic trong code. Chúng không đến từ mô hình VideoMAE và không được tối ưu bằng bộ correct/incorrect.

Nếu dưới 3 cặp, feedback trả thông báo chưa đủ dữ liệu. Những cặp được gọi stable trong comment chỉ là nhóm có DTW nhỏ hơn; chưa được chuyên gia xác nhận hợp lệ.

### 13.3. Áp dụng vào clip của bạn

| Vùng | Distance | Threshold | Kết quả số học |
| --- | --- | --- | --- |
| Tay trái | 0.8609 | 0.6781 | Vượt ngưỡng -> needs_attention |
| Tay phải | 1.0039 | 1.3547 | Dưới ngưỡng -> ok |
| Mặt | 0.4928 | 1.6311 | Dưới ngưỡng -> ok |

Quy tắc là distance / max(threshold, 1e-8) > 1 thì needs_attention.

Feedback ok chỉ có nghĩa phép đo chưa vượt ngưỡng. Do mask chưa tham gia vào phép đo vùng, kết luận tay phải ok dù chỉ được phát hiện 10/54 frame là chưa đủ căn cứ để xác nhận kỹ thuật tay phải.

Tương tự, face ok không chứng minh biểu cảm ngôn ngữ ký hiệu đúng, vì chỉ so sáu anchor vị trí.

### 13.4. Đoạn 0-17% đến từ đâu?

Lỗi vùng chung lấy median top-3, nhưng trace tìm đoạn lỗi hiện chỉ lấy của reference gần nhất.

Hàm largest_error_segment tìm đoạn LIÊN TỤC DÀI NHẤT có trace > threshold. Nó không chọn đoạn có giá trị lỗi lớn nhất, tổng lỗi lớn nhất hoặc timestamp quan trọng nhất.

Với clip 54 frame:

~~~text
start_frame = 0
end_frame = 9

start_percent = round(100 * 0 / (54-1)) = 0
end_percent   = round(100 * 9 / (54-1)) = 17
~~~

Frame 0..9 là 10 frame đầu của clip đã trim. % dựa trên vị trí frame, không phải timestamp thật. Những index này không phải index của original_video.mp4.

Câu hiện tại viết lệch rõ nhất là quá mạnh so với phép đo đoạn dài nhất vượt ngưỡng. Đúng hơn là vùng này có một đoạn đầu vượt ngưỡng liên tục so với reference gần nhất.

Code chưa biết bạn nên nâng tay bao nhiêu, đổi hướng nào hoặc sửa ngón tay nào. Không nên tự suy ra cách sửa cụ thể chỉ từ câu feedback này.

## 14. Kết luận cuối cùng tính thế nào?

Nguồn: [sign_eval/evaluator.py](C:/HandSign/sign_eval/evaluator.py:372).

| Điều kiện | status tổng |
| --- | --- |
| Tracking invalid hoặc low_quality | not_scored |
| Tracking qua và recognition mismatch | incorrect_label |
| Tracking qua, recognition match, form excellent/good | correct |
| Tracking qua, recognition match, form trạng thái khác | right_label_needs_practice |
| Tracking qua, recognition uncertain/not_run | needs_label_confirmation |

Với ví dụ của bạn:

~~~text
48/54 frame có ít nhất một tay -> tracking ok
VideoMAE nhận Cảm ơn, confidence/margin vượt ngưỡng -> match
D_user = 6.61395 > 4.94003 -> form needs_practice
match + needs_practice -> right_label_needs_practice
~~~

score=3.03 được đính kèm để mô tả thứ hạng reference. Nó không trực tiếp quyết định form cho Cảm ơn khi validation đang có.

Chưa có trạng thái cần quay lại cho từng tay riêng, chưa có ngưỡng semantic đã được hiệu chỉnh và chưa kết hợp uncertainty của nhiều nhánh vào một xác suất cuối.

## 15. Giải nghĩa toàn bộ JSON

| Field | Nguồn / phép tính |
| --- | --- |
| target_label | Label từ CLI đã khớp với tên chuẩn trong calibration |
| sequence_path | Đường dẫn sequence evaluator nhận |
| tracking.status | All-zero, có mask hay không, và ngưỡng any-hand 70% |
| tracking.frame_count | T của sequence sau trim |
| tracking.*_detected_fraction | Trung bình boolean mask tương ứng |
| recognition.status | So predicted_label và target sau kiểm tra confidence/margin |
| recognition.predicted_label | model.config.id2label của lớp top-1 |
| recognition.confidence | Softmax top-1, hiển thị làm tròn |
| recognition.margin | Top-1 trừ top-2, hiển thị làm tròn |
| status | Bảng tổng hợp tracking + recognition + form ở mục 14 |
| form.status | Validation threshold nếu có; nếu không là bảng percentile |
| form.score | 100 * count(b_i >= D_user) / n |
| form.comparison_distance | Mean của top-k DTW nhỏ nhất |
| form.reference_count | Số reference được ghi trong calibration |
| form.calibration_count | Số giá trị leave-one-out dùng để xếp hạng |
| form.calibration_quality | Đủ/thiếu 10 reference |
| form.decision_source | labeled_user_validation hoặc reference_distribution |
| form.validation | Mục calibration user của label được chép vào output |
| feedback[].region | left_hand, right_hand hoặc face |
| feedback[].distance | Median lỗi vùng từ tối đa 3 reference gần nhất |
| feedback[].threshold | Quantile 90% lỗi vùng từ các cặp reference được chọn |
| feedback[].severity | So distance với threshold |
| feedback[].segment | Đoạn dài nhất vượt ngưỡng trên trace của best reference |
| feedback[].message | Mẫu câu viết sẵn được điền tên vùng và % |
| feedback_calibration.thresholds | Các threshold vùng trước khi rút gọn về 4 chữ số |
| feedback_calibration.sample_count | Số cặp dùng tính threshold vùng |
| nearest_references | Tên file và DTW của top-k reference gần nhất |

recognizer.predict_video() tạo cả top_predictions, nhưng evaluator hiện chỉ giữ top-1/confidence/margin trong nhóm recognition. Vì vậy top-5 nhãn không xuất hiện đầy đủ trong evaluation.json.

Dòng Saved evaluation to ... là log console sau khi json.dump hoàn tất, không phải một field của JSON.

## 16. Các hạn chế và lời giải thích trước cần đính chính

| Nội dung | Trạng thái thực tế |
| --- | --- |
| Đã xử lý missing landmark bằng mask | Mới lưu mask và chặn tracking toàn clip. Normalize/DTW/feedback vẫn dùng cả giá trị missing |
| Đã normalize theo cơ thể hoặc mắt | Chưa làm; vẫn z-score toàn clip |
| Đã nội suy, resample và đo lỗi tốc độ | Chưa làm |
| Chỉ chấm form khi nhãn đúng | Chưa đúng: vẫn chấm khi mismatch/uncertain, sau đó đặt status |
| Model lỗi thì tự chạy landmark-only | Chưa có fallback trong CLI khi --video lỗi |
| Đã chống self-match | Mới bỏ demo mặc định; runtime vẫn nhận chính reference làm user |
| Score thấp nghĩa là phần trăm động tác đúng thấp | Không: đây là percentile của reference |
| 90% nghĩa là đã kiểm chứng trên user mới | Không: đây là kết quả fit trên cùng 10 clip |
| Tay/mặt ok chứng minh làm đúng | Chỉ là dưới ngưỡng số học; tracking và feature còn hạn chế |
| Đoạn feedback là đoạn sai nhất | Code chọn đoạn dài nhất vượt ngưỡng của best reference |
| Chạy builder là cập nhật đầy đủ mọi mẫu mới | Chưa có tự rebuild cache info.npy |
| Đã dọn sạch cấu trúc project | Mới gom core code; archive, dữ liệu cũ, notebook và alias vẫn còn |

Mình đã mô tả một số cải tiến như thể đã hoàn tất trong các lượt trước. Phần đối chiếu này sửa lại các mô tả đó theo code đang chạy.

Có thêm các điểm cần lưu ý cụ thể cho việc phát triển:

- Paths mặc định vẫn tương đối với thư mục chạy lệnh; nên chạy ở C:\HandSign.
- evaluate.py ưu tiên record.npz nếu tồn tại, dù sequence.npy có thể được thay mới riêng. Chưa kiểm tra hai file có cùng attempt.
- Bạn có thể truyền video của lần quay A và sequence của lần B. CLI chưa kiểm tra tính đồng nhất.
- Các artifact calibration chưa có dấu vết version feature/DTW để ngăn dùng nhầm sau khi đổi normalize.
- Không có kiểm tra chuyển động thực sự; có tay trong ảnh không đồng nghĩa đang thực hiện ký hiệu.
- Trọng số vùng, độ quan trọng một/two-hand và lỗi ngón tay chưa được mô hình hóa theo label.
- Low-sample chỉ được ghi ra, chưa có cơ chế bảo thủ đầy đủ để chặn một phán quyết mạnh.
- Median đúng và threshold được lưu làm tròn, có thể ảnh hưởng ranh rất sát.
- Bộ cài requirements có khoảng version rộng. Chưa kiểm tra cài mới sạch để bảo đảm mọi tổ hợp version tương thích.

## 17. Thư viện và các kiểm tra đã làm

requirements.txt hiện dùng các khoảng version:

~~~text
numpy>=1.26,<3
fastdtw>=0.3.4
mediapipe>=0.10.14
opencv-python>=4.8
torch>=2.2
transformers>=4.40
huggingface-hub>=0.23
decord>=0.6.0
~~~

Vai trò: NumPy xử lý mảng và thống kê; FastDTW ghép chuỗi; MediaPipe lấy landmark; OpenCV quay/ghi/video UI; Torch chạy model; Transformers cung cấp VideoMAE và processor; Hugging Face Hub lấy checkpoint; Decord đọc frame video.

Torch và các thư viện nhận diện được import bên trong hàm khi dùng model. Tuy nhiên chúng vẫn nằm trong cùng requirements.txt, nên pip install -r requirements.txt sẽ cài cả nhóm đó. Chữ optional trong comment mô tả đường chạy, chưa tách thành gói cài optional riêng.

Phiên bản gói được đọc trên môi trường hiện tại: NumPy 1.26.4, FastDTW 0.3.4, MediaPipe 1.0.1, OpenCV 4.11.0.86, Torch 2.9.1, Transformers 4.57.1, Hugging Face Hub 0.36.2, Decord 0.6.0. Đây là metadata gói trên máy, không phải kết quả kiểm thử mọi API của từng gói.

Trong đợt sửa trước đã chạy compile và năm unit test:

1. Z-score đưa mean về khoảng 0 và std về khoảng 1.
2. Trim landmarks và mask cùng khoảng.
3. DTW của hai sequence giống nhau bằng 0.
4. Tracking từ chối sequence có mask thiếu tay.
5. Chọn được threshold đúng trên dữ liệu synthetic tách biệt.

Năm test này kiểm tra hàm nền. Chúng chưa kiểm chứng độ chính xác nhận diện, tính đúng ngôn ngữ ký hiệu, matching checkpoint, timestamp camera, dữ liệu một tay hoặc chất lượng lời feedback.

Trong lượt giải thích này mình đã:

- Đọc lại code và hai calibration JSON.
- Đọc record.npz: đúng 54 frame, mask tay trái 38, tay phải 10, mặt 54.
- Đọc metadata video: 106 frame gốc và 54 frame trim, 640x480, FPS 30.
- Tính lại form và feedback từ NPZ; kết quả khớp JSON bạn gửi.
- Dùng recognition từ JSON đã có làm đầu vào để tái hiện kết luận, không chạy lại VideoMAE trong lượt này.
- Tính lại D của 10 clip correct/incorrect.
- Kiểm tra ba cặp cache so với DTW hiện tại.
- Thử self-match và xác nhận lỗ hổng vẫn còn.
- Đọc cấu hình processor/model đã cache để xác nhận thông số tiền xử lý.

## 18. Cách sử dụng các file đúng vai trò

Chuẩn bị calibration reference, khi cache reference đã phù hợp:

~~~powershell
python build_calibration.py
~~~

Fit ngưỡng Cảm ơn từ hai folder bạn đã gán nhãn:

~~~powershell
python build_validation.py --label "Cảm ơn" --correct-dir data/eval_set/correct --incorrect-dir data/eval_set/incorrect
~~~

Hai lệnh trên không cần chạy lại cho mỗi lần user luyện tập. Fit validation lại khi thay tập clip có nhãn hoặc thay chuẩn khoảng cách/reference.

Quay một lần mới:

~~~powershell
python record.py
~~~

Nhận diện và đánh giá cùng lần quay đó:

~~~powershell
python evaluate.py --target-label "Cảm ơn" --sequence data/user/sequence_user/record.npz --video data/user/video_user/trimmed_video.mp4
~~~

Chỉ đánh giá landmark:

~~~powershell
python evaluate.py --target-label "Cảm ơn" --sequence data/user/sequence_user/record.npz
~~~

Lệnh cuối vẫn tính form nhưng không xác nhận label bằng video. Baseline.py là alias; không cần chạy thêm sau build_calibration.py.

Để tiến tới một hệ thống chấm kỹ thuật đáng tin cậy, các phần còn thiếu đã được ghi ở mục 16 cần được xử lý và kiểm chứng bằng clip độc lập. Tài liệu này phản ánh phiên bản đang có để bạn có thể giải thích và kiểm tra từng quyết định, không coi các phép đo thử nghiệm là kết luận chuyên môn đã được xác thực.
