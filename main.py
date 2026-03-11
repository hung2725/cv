import cv2
from detection import TrafficDetector
from traffic_violation import ViolationChecker

def run_system():
    detector = TrafficDetector()
    # Chỉnh con số này để vạch đỏ đè đúng vạch sơn trắng trong video
    checker = ViolationChecker(stop_line_y=380) 
    
    cap = cv2.VideoCapture("16h15.15.9.22.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Xử lý nhận diện và tracking
        detections, lights = detector.detect_all(frame)
        # Xử lý vi phạm và vẽ giao diện
        result_frame = checker.process(frame, detections, lights)

        cv2.imshow("Giam Sat Giao Thong - YOLO 2026", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()