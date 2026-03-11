import cv2
import os
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.vehicle_model = YOLO(os.path.join(BASE_DIR, 'models', 'vehicle.pt'))
        self.light_model = YOLO(os.path.join(BASE_DIR, 'models', 'traffic_light.pt'))

    def detect_all(self, frame):
        # Tracking xe để giữ ID cố định
        v_res = self.vehicle_model.track(frame, persist=True, imgsz=480, verbose=False)[0]
        detections = []
        if v_res.boxes.id is not None:
            ids = v_res.boxes.id.int().cpu().tolist()
            bboxes = v_res.boxes.xyxy.int().cpu().tolist()
            clss = v_res.boxes.cls.int().cpu().tolist()
            for box, id, cls in zip(bboxes, ids, clss):
                detections.append({"box": box, "id": id, "type": self.vehicle_model.names[cls]})

        # Nhận diện cột đèn và đóng khung
        l_res = self.light_model(frame, imgsz=320, verbose=False)[0]
        lights = []
        for l_box in l_res.boxes:
            lx1, ly1, lx2, ly2 = map(int, l_box.xyxy[0])
            status = self.light_model.names[int(l_box.cls[0])]
            lights.append({"box": [lx1, ly1, lx2, ly2], "status": status})
        
        return detections, lights