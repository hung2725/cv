# Hệ thống phát hiện vi phạm giao thông (YOLO)

### 1. Giới thiệu

Dự án này xây dựng một **hệ thống giám sát giao thông tự động** từ video, có thể:
- Phát hiện các phương tiện tham gia giao thông (xe máy, ô tô, ...).
- Nhận diện **trạng thái đèn giao thông** (đỏ/xanh/...).
- Kiểm tra **vi phạm vượt đèn đỏ**.
- Kiểm tra **đội mũ bảo hiểm** đối với xe máy.
- Lưu lại **ảnh bằng chứng** và phát âm thanh cảnh báo khi phát hiện vi phạm.

Toàn bộ pipeline được hiện thực bằng Python và các thư viện thị giác máy tính/deep learning hiện đại.

---

### 2. Công nghệ sử dụng

- **Ngôn ngữ**: Python 3

- **Deep Learning / Object Detection**
  - `ultralytics` (YOLO) để phát hiện:
    - Phương tiện giao thông (`vehicle.pt`).
    - Đèn giao thông (`traffic_light.pt`).
    - Mũ bảo hiểm (`helmet.pt`).
  - `PyTorch` là backend cho YOLO (được tích hợp sẵn trong `ultralytics`).

- **Xử lý ảnh / video**
  - `OpenCV` (`cv2`):
    - Đọc video đầu vào (`VideoCapture`).
    - Vẽ bounding box, text, vạch dừng ảo.
    - Hiển thị frame kết quả lên màn hình.

- **OCR / nhận dạng ký tự (tiềm năng mở rộng)**
  - `easyocr`, `pytesseract` (có trong `requirements.txt`) để đọc biển số xe, phù hợp với model `plate` trong phần huấn luyện (nếu triển khai).

- **Khác**
  - `numpy` cho xử lý ma trận/ảnh.
  - `winsound` (trên Windows) để phát âm thanh cảnh báo khi có vi phạm.

Xem chi tiết danh sách phụ thuộc trong `requirements.txt`.

---

### 3. Cách hệ thống hoạt động

#### 3.1. Luồng xử lý chính (runtime)

File chính: `main.py`

1. **Khởi tạo mô hình**
   - Tạo đối tượng `TrafficDetector` (trong `detection.py`) để:
     - Load các model YOLO: `vehicle.pt`, `traffic_light.pt`, `helmet.pt` từ thư mục `models/`.
   - Tạo đối tượng `ViolationChecker` (trong `traffic_violation.py`) với tham số:
     - `stop_line_y`: tọa độ y của **vạch dừng ảo** trên khung hình.
     - Giá trị này cần chỉnh để **trùng với vạch sơn thực tế** trong video (ví dụ: 550, 580, 600, ...).

2. **Đọc video đầu vào**
   - Dùng `cv2.VideoCapture("16h15.15.9.22.mp4")` để mở file video.

3. **Xử lý từng frame**
   - Với mỗi frame lấy được từ video:
     1. Gọi `detector.detect_all(frame)`:
        - **Stage phát hiện xe**:
          - YOLO `vehicle.pt` phát hiện các phương tiện.
          - Kết quả mỗi xe gồm: `box = [x1, y1, x2, y2]`, `type` (loại xe), `has_helmet` (ban đầu `None`).
        - **Kiểm tra mũ bảo hiểm cho xe máy**:
          - Nếu `type == 'motorbike'`, cắt vùng ROI quanh người lái.
          - Chạy model `helmet.pt` trên ROI để xem có mũ bảo hiểm hay không.
          - Gán `has_helmet = 'no_helmet'` nếu phát hiện không đội mũ.
        - **Stage nhận diện đèn giao thông**:
          - YOLO `traffic_light.pt` phát hiện đèn và trả về `status` (`red`, `green`, ...).
        - Hàm trả về:
          - `detections`: danh sách xe + bounding box + trạng thái mũ bảo hiểm.
          - `lights`: danh sách đèn + bounding box + `status`.

     2. Gọi `checker.process(frame, detections, lights)`:
        - Lấy trạng thái đèn hiện tại (`current_status`), ví dụ `"red"` hoặc `"green"`.
        - Vẽ **vạch dừng ảo** ngang khung hình tại `stop_line_y`:
          - Màu **đỏ** nếu đèn đang đỏ.
          - Màu **xanh lá** nếu đèn không đỏ.
        - Vẽ text trạng thái đèn: `LIGHT: RED` hoặc tương tự ở góc màn hình.

        - Với mỗi phương tiện trong `detections`:
          - Lấy vị trí bounding box `(x1, y1, x2, y2)`.

          **a. Kiểm tra vượt đèn đỏ**
          - Nếu:
            - `current_status.lower() == "red"` (đèn đỏ) **và**
            - `y2 > stop_line_y - 5` (đuôi xe vượt qua vạch dừng),
          - Thì:
            - Vẽ khung đỏ quanh xe
            - Ghi text: `"VI PHAM: VUOT DEN DO"`.
            - Lưu **ảnh bằng chứng** vào thư mục `evidence/` với tên dạng `red_light_HHMMSS_xxxxxx.jpg`.
            - Phát tiếng **beep** bằng `winsound.Beep(...)`.

          **b. Kiểm tra mũ bảo hiểm**
          - Nếu `has_helmet == 'no_helmet'`:
            - Ghi text `"NO HELMET"` phía trên xe để cảnh báo.

        - Hàm trả về `frame` đã được vẽ đầy đủ thông tin (vạch, bbox, text, ...).

     3. Hiển thị frame kết quả:
        - Dùng `cv2.imshow(...)` để xem trực tiếp.
        - Nhấn phím `q` để thoát vòng lặp.

4. **Kết thúc**
   - Giải phóng video (`cap.release()`).
   - Đóng toàn bộ cửa sổ OpenCV (`cv2.destroyAllWindows()`).

#### 3.2. Tóm tắt luồng xử lý

- **Input**: Video giao thông (`16h15.15.9.22.mp4`).
- **Xử lý**:
  - YOLO phát hiện xe + đèn + mũ bảo hiểm.
  - Kiểm tra vị trí xe so với vạch dừng khi đèn đỏ.
  - Kiểm tra trạng thái đội mũ bảo hiểm.
- **Output**:
  - Video hiển thị real-time với bounding box và cảnh báo.
  - Thư mục `evidence/` chứa ảnh chụp lại các trường hợp vi phạm.
  - Âm thanh cảnh báo khi có vi phạm.

---

### 4. Huấn luyện mô hình (train_all.py)

File `train_all.py` dùng để **huấn luyện lại các model YOLO**:

- Gồm danh sách 4 task:
  - `vehicle_model`: dữ liệu trong `data/vehicle_data/vehicle.yaml`.
  - `light_model`: dữ liệu trong `data/traffic_light_data/traffic_light.yaml`.
  - `plate_model`: dữ liệu trong `data/plate_data/plate.yaml`.
  - `helmet_model`: dữ liệu trong `data/helmet_data/helmet.yaml`.

- Với mỗi task:
  - Khởi tạo `YOLO("yolo11n.pt")` (model base).
  - Gọi `model.train(...)` với:
    - `data`: đường dẫn file YAML.
    - `epochs`: số epoch (ví dụ 100).
    - `imgsz`: kích thước ảnh (640 hoặc 416).
    - `batch`: batch size.
    - `device=0`: dùng GPU (CUDA).
    - `name`: tên experiment (lưu trong `runs/detect/`).

Sau khi train xong, bạn có thể copy các file `.pt` đã train vào thư mục `models/` để dùng trong `detection.py`.

---

### 5. Cách chạy nhanh

#### 5.1. Cài đặt môi trường

1. Tạo và kích hoạt môi trường ảo (khuyến khích):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```

3. Chuẩn bị model:
   - Tạo thư mục `models/` (nếu chưa có).
   - Đặt các file:
     - `vehicle.pt`
     - `traffic_light.pt`
     - `helmet.pt`
     - (tuỳ chọn) `plate.pt`
   - vào trong thư mục `models/` cùng cấp với các file `.py`.

4. Đặt video test:
   - Đảm bảo file `16h15.15.9.22.mp4` nằm cùng thư mục với `main.py`, hoặc sửa lại đường dẫn trong code.

#### 5.2. Chạy chương trình

```bash
python main.py
```

- Một cửa sổ sẽ bật lên hiển thị video kèm:
  - Vạch dừng ảo.
  - Trạng thái đèn.
  - Bounding box quanh xe.
  - Cảnh báo `"VI PHAM: VUOT DEN DO"` và/hoặc `"NO HELMET"` nếu phát hiện vi phạm.
- Nhấn phím `q` để thoát chương trình.

---

### 6. Ghi chú

- **Hiệu chỉnh vạch dừng**:
  - Tham số `stop_line_y` trong `main.py` / `ViolationChecker` rất quan trọng.
  - Cần điều chỉnh cho trùng vị trí vạch trên từng video khác nhau.

- **Hiệu năng**:
  - Nên chạy trên máy có GPU (RTX) để tốc độ real-time mượt hơn.
  - Trong `detection.py` có thể tinh chỉnh tham số `imgsz`, `stream` để tối ưu.

- **Mở rộng**:
  - Có thể kết hợp module đọc biển số (plate) để:
    - Tự động trích xuất biển số từ các frame vi phạm.
    - Lưu kèm text biển số vào log/báo cáo.

