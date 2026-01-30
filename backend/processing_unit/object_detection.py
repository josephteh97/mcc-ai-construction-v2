from ultralytics import YOLO
import cv2
import os
import numpy as np
from typing import List, Dict, Any

class ObjectDetector:
    def __init__(self, model_path: str = "yolo26n.pt"):
        """
        Initialize the YOLOv11 detector.
        
        Args:
            model_path (str): Path to the trained .pt file. 
                              Defaults to standard pre-trained weights if custom model not available.
        """
        self.model_path = model_path
        # Try to load custom trained model first
        if os.path.exists(self.model_path):
            print(f"[INFO] Loading Custom YOLO Model from: {os.path.abspath(self.model_path)}")
            self.model = YOLO(self.model_path)
        else:
            print(f"[WARN] Custom model not found at {self.model_path}. Falling back to 'yolo11n.pt' (COCO pretrained).")
            print("[WARN] Structural elements like columns/beams will NOT be detected correctly.")
            self.model = YOLO("yolo11n.pt") 
            
        print(f"[INFO] YOLO Model Classes: {self.model.names}")
        self.class_names = self.model.names

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Run inference on an image.
        
        Args:
            image_path (str): Path to the input image.
            conf_threshold (float): Confidence threshold for detections.
            
        Returns:
            Dict containing detections and metadata.
        """
        results = self.model.predict(image_path, conf=conf_threshold)
        result = results[0] # We process one image at a time
        
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = self.class_names[cls_id]
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            
            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": xyxy
            })
            
        return {
            "file": image_path,
            "count": len(detections),
            "detections": detections
        }

    def visualize(self, image_path: str, output_path: str):
        """
        Visualize detections and save the image.
        """
        results = self.model.predict(image_path)
        result = results[0]
        result.save(filename=output_path)  # save to disk
