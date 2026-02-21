import cv2
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

class QualityGate:
    def __init__(self, blur_threshold=100, brightness_threshold=40, log_path="docs/risk_management/quality_logs.json"):
        self.blur_threshold = blur_threshold
        self.brightness_threshold = brightness_threshold
        self.log_path = Path(log_path)
        
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not self.log_path.exists():
            with open(self.log_path, 'w') as f:
                json.dump([], f)

    def _log_rejection(self, image_path, metrics, reason):
        """Log rejected image details for ISO 14971 compliance."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "image_id": os.path.basename(image_path),
            "status": "REJECTED",
            "reason": reason,
            "metrics": metrics
        }
        
        try:
            with open(self.log_path, 'r+') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
                
                logs.append(entry)
                f.seek(0)
                json.dump(logs, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to log rejection: {e}")

    def is_gradable(self, image_path):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                self._log_rejection(image_path, {}, "Image load failed")
                return False, {"error": "Image load failed"}
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Check for Blur (Laplacian Variance)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Check for Brightness
            avg_brightness = np.mean(gray)
            
            metrics = {
                "blur_score": float(variance), 
                "brightness": float(avg_brightness)
            }
            
            # Decision Logic
            if variance < self.blur_threshold:
                self._log_rejection(image_path, metrics, f"Blur score {variance:.1f} < {self.blur_threshold}")
                return False, metrics
                
            if avg_brightness < self.brightness_threshold:
                self._log_rejection(image_path, metrics, f"Brightness {avg_brightness:.1f} < {self.brightness_threshold}")
                return False, metrics
            
            return True, metrics
            
        except Exception as e:
            self._log_rejection(image_path, {}, f"Processing error: {str(e)}")
            return False, {"error": str(e)}
