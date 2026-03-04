import cv2
from detection import TrafficDetector
from traffic_violation import ViolationChecker

def run_system():
    detector = TrafficDetector()
    # Chỉnh lại tọa độ y để vạch đỏ khớp với vạch trắng trên đường
    checker = ViolationChecker(stop_line_y=700) 
    
    cap = cv2.VideoCapture("16h15.15.9.22.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30  # Dự phòng nếu không lấy được FPS

    output_filename = "output_violation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"Bắt đầu xử lý. Video sẽ được lưu tại: {output_filename}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Xử lý nhận diện
        detections, lights = detector.detect_all(frame)
        result_frame = checker.process(frame, detections, lights)

        # Ghi khung hình đã xử lý vào file video
        out.write(result_frame)

        # Hiển thị
        cv2.imshow("may farm tien tu dong", result_frame)
        
        # Nhấn Space để thoát
        if cv2.waitKey(1) & 0xFF == ord(' '): break

    # Giải phóng tài nguyên
    cap.release()
    out.release() # Quan trọng: Phải đóng out thì file video mới xem được
    cv2.destroyAllWindows()
    print("Đã lưu video thành công!")

if __name__ == "__main__":
    run_system()