import cv2
import datetime
import os

class ViolationChecker:
    def __init__(self, stop_line_y=400):
        self.stop_line_y = stop_line_y
        # Vùng giới hạn: chỉ xét xe xuất hiện từ phía dưới vạch dừng (vùng chờ)
        self.roi_y_start = stop_line_y + 20 

    def process(self, frame, detections, lights):
        current_status = str(lights[0]['status']).lower() if lights else "none"
        is_red = any(x in current_status for x in ['red', 'do', '0'])
        
        # Vẽ vạch dừng và vùng kiểm soát
        color = (0, 0, 255) if is_red else (0, 255, 0)
        cv2.line(frame, (0, self.stop_line_y), (frame.shape[1], self.stop_line_y), color, 3)
        
        for v in detections:
            x1, y1, x2, y2 = v['box']
            plate_text = v.get("plate_text", "")

            # LOGIC QUAN TRỌNG:
            # 1. Xe phải đi từ dưới lên (y2 > stop_line_y lúc bắt đầu)
            # 2. Đèn đỏ và xe vượt lên phía trên vạch (y2 < stop_line_y)
            # Xe hướng khác đi ngang sẽ có y luôn < stop_line_y hoặc nằm ngoài vùng này
            
            if is_red and y2 < self.stop_line_y:
                # Kiểm tra thêm: Nếu xe này không phải xe từ hướng chờ (ví dụ x nằm quá xa bên trái/phải) 
                # thì bạn có thể thêm điều kiện tọa độ x tại đây.
                
                # Hiển thị biển số và khung vi phạm trực tiếp trên video
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                if plate_text:
                    cv2.rectangle(frame, (x1, y1 - 65), (x1 + 250, y1 - 35), (0, 0, 0), -1)
                    cv2.putText(frame, f"BS: {plate_text}", (x1 + 5, y1 - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "VUOT DEN DO", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame 