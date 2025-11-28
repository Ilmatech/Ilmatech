from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the Object Detector with YOLOv8.
        
        Args:
            model_path (str): Path to the YOLOv8 model file.
        """
        self.model = YOLO(model_path)

    def detect(self, frame, conf_threshold=0.5, classes=None):
        """
        Detect objects in the frame.
        
        Args:
            frame (numpy.ndarray): Input frame.
            conf_threshold (float): Confidence threshold for detections.
            classes (list): List of class IDs to filter.
            
        Returns:
            list: List of detections [x1, y1, x2, y2, score, class_id]
        """
        results = self.model(frame, conf=conf_threshold, classes=classes, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                detections.append([x1, y1, x2, y2, score, class_id])
                
        return detections
