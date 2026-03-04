import cv2
import os
from ultralytics import YOLO
import easyocr

class TrafficDetector:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.vehicle_model = YOLO(os.path.join(BASE_DIR, 'models', 'vehicle.pt'))
        self.light_model = YOLO(os.path.join(BASE_DIR, 'models', 'traffic_light.pt'))
        self.plate_model = YOLO(os.path.join(BASE_DIR, 'models', 'license_plate.pt'))
        # Tận dụng RTX 4050 để đọc biển số nhanh
        self.reader = easyocr.Reader(['en'], gpu=True)

    def detect_all(self, frame):
        v_res = self.vehicle_model(frame, imgsz=640, verbose=False)[0]
        detections = []
        for box in v_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            v_data = {"box": [x1, y1, x2, y2], "plate_text": "", "plate_box": None}
            # Tìm biển số
            roi_v = frame[y1:y2, x1:x2]
            if roi_v.size > 0:
                p_res = self.plate_model(roi_v, imgsz=416, verbose=False)[0]
                if len(p_res.boxes) > 0:
                    px1, py1, px2, py2 = map(int, p_res.boxes[0].xyxy[0])
                    v_data["plate_box"] = [x1 + px1, y1 + py1, x1 + px2, y1 + py2]
                    # OCR biển số
                    plate_crop = frame[y1+py1:y1+py2, x1+px1:x1+px2]
                    if plate_crop.size > 0:
                        ocr_res = self.reader.readtext(plate_crop, detail=0)
                        if ocr_res:
                            v_data["plate_text"] = "".join(ocr_res).upper().replace(" ", "")
            detections.append(v_data)
        l_res = self.light_model(frame, imgsz=416, verbose=False)[0]
        lights = [{"status": self.light_model.names[int(l.cls[0])]} for l in l_res.boxes]
        return detections, lights