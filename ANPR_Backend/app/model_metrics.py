import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMetricsTracker:
    def __init__(self, metrics_file="model_metrics.json"):
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()

    def _load_metrics(self):
        """Load existing metrics from file or create new structure"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics file: {e}")
                return self._create_new_metrics()
        return self._create_new_metrics()

    def _create_new_metrics(self):
        """Create a new metrics structure"""
        return {
            "statistics": {
                "total_images": 0,
                "total_detections": 0,
                "models": {
                    "yolov5s": {
                        "detection_rate": 0.0,
                        "avg_confidence": 0.0,
                        "max_confidence": 0.0,
                        "min_confidence": 1.0
                    },
                    "yolov8s": {
                        "detection_rate": 0.0,
                        "avg_confidence": 0.0,
                        "max_confidence": 0.0,
                        "min_confidence": 1.0
                    },
                    "yolov11s": {
                        "detection_rate": 0.0,
                        "avg_confidence": 0.0,
                        "max_confidence": 0.0,
                        "min_confidence": 1.0
                    },
                    "pytorch": {
                        "detection_rate": 0.0,
                        "avg_confidence": 0.0,
                        "max_confidence": 0.0,
                        "min_confidence": 1.0
                    },
                    "kaggle": {
                        "detection_rate": 0.0,
                        "avg_confidence": 0.0,
                        "max_confidence": 0.0,
                        "min_confidence": 1.0
                    },
                    "keras": {
                        "detection_rate": 0.0,
                        "avg_confidence": 0.0,
                        "max_confidence": 0.0,
                        "min_confidence": 1.0
                    }
                }
            },
            "recent_detections": []
        }

    def _save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metrics file: {e}")

    def add_detection(self, image_name, detection_results):
        """Add a new detection result with all available models
        
        Args:
            image_name: Name of the image
            detection_results: Dictionary containing detection results from all models
        """
        timestamp = datetime.now().isoformat()
        
        # Extract all available models
        all_models = []
        detections_dict = {}
        
        # Check for YOLOv5s
        if "yolov5s_detections" in detection_results:
            yolov5s_detected = len(detection_results["yolov5s_detections"]) > 0
            yolov5s_confidence = max([d.get('confidence', 0) for d in detection_results["yolov5s_detections"]]) if yolov5s_detected else 0
            all_models.append("yolov5s")
            detections_dict["yolov5s"] = {"detected": yolov5s_detected, "confidence": yolov5s_confidence}
        
        # Check for YOLOv8s
        if "yolov8s_detections" in detection_results:
            yolov8s_detected = len(detection_results["yolov8s_detections"]) > 0
            yolov8s_confidence = max([d.get('confidence', 0) for d in detection_results["yolov8s_detections"]]) if yolov8s_detected else 0
            all_models.append("yolov8s")
            detections_dict["yolov8s"] = {"detected": yolov8s_detected, "confidence": yolov8s_confidence}
        
        # Check for YOLOv9s
        if "yolov9s_detections" in detection_results:
            yolov9s_detected = len(detection_results["yolov9s_detections"]) > 0
            yolov9s_confidence = max([d.get('confidence', 0) for d in detection_results["yolov9s_detections"]]) if yolov9s_detected else 0
            all_models.append("yolov9s")
            detections_dict["yolov9s"] = {"detected": yolov9s_detected, "confidence": yolov9s_confidence}
        
        # Check for YOLOv11s
        if "yolov11s_detections" in detection_results:
            yolov11s_detected = len(detection_results["yolov11s_detections"]) > 0
            yolov11s_confidence = max([d.get('confidence', 0) for d in detection_results["yolov11s_detections"]]) if yolov11s_detected else 0
            all_models.append("yolov11s")
            detections_dict["yolov11s"] = {"detected": yolov11s_detected, "confidence": yolov11s_confidence}
        
        # Check for PyTorch
        if "pytorch_detections" in detection_results:
            pytorch_detected = len(detection_results["pytorch_detections"]) > 0
            pytorch_confidence = max([d.get('confidence', 0) for d in detection_results["pytorch_detections"]]) if pytorch_detected else 0
            all_models.append("pytorch")
            detections_dict["pytorch"] = {"detected": pytorch_detected, "confidence": pytorch_confidence}
        
        # Check for Kaggle model
        if "kaggle_detections" in detection_results:
            kaggle_detected = len(detection_results["kaggle_detections"]) > 0
            kaggle_confidence = max([d.get('confidence', 0) for d in detection_results["kaggle_detections"]]) if kaggle_detected else 0
            all_models.append("kaggle")
            detections_dict["kaggle"] = {"detected": kaggle_detected, "confidence": kaggle_confidence}
            
            # Add weapon type if available
            if kaggle_detected and "model_statistics" in detection_results and "kaggle" in detection_results["model_statistics"]:
                weapon_type = detection_results["model_statistics"]["kaggle"].get("weapon_type", "Unknown")
                detections_dict["kaggle"]["weapon_type"] = weapon_type
        
        # Check for Keras
        if "keras_detections" in detection_results:
            keras_detected = len(detection_results["keras_detections"]) > 0
            keras_confidence = max([d.get('confidence', 0) for d in detection_results["keras_detections"]]) if keras_detected else 0
            all_models.append("keras")
            detections_dict["keras"] = {"detected": keras_detected, "confidence": keras_confidence}
        
        # Calculate model agreement
        active_detections = [detections_dict[model]["detected"] for model in all_models]
        agreement_count = sum(active_detections)
        agreement_percentage = (agreement_count / len(all_models)) * 100 if all_models else 0
        
        # Create detection record
        detection = {
            "timestamp": timestamp,
            "image_name": image_name,
            "models": detections_dict,
            "agreement": {
                "count": agreement_count,
                "total_models": len(all_models),
                "percentage": agreement_percentage,
                "all_agree": len(all_models) > 0 and (agreement_count == len(all_models) or agreement_count == 0),
                "majority_agree": agreement_count > (len(all_models) / 2) if len(all_models) > 0 else False,
                "any_detected": any(active_detections)
            }
        }
        
        # Add all detection results to the record
        for model in all_models:
            detection[f"{model}_detections"] = detection_results.get(f"{model}_detections", [])

        # Update recent detections (keep last 100)
        self.metrics["recent_detections"].insert(0, detection)
        self.metrics["recent_detections"] = self.metrics["recent_detections"][:100]

        # Update statistics
        self.metrics["statistics"]["total_images"] += 1
        if any(active_detections):
            self.metrics["statistics"]["total_detections"] += 1

        # Update model-specific statistics for all available models
        for model in all_models:
            if model not in self.metrics["statistics"]["models"]:
                self.metrics["statistics"]["models"][model] = {
                    "detection_rate": 0.0, 
                    "avg_confidence": 0.0, 
                    "max_confidence": 0.0, 
                    "min_confidence": 1.0
                }
            
            stats = self.metrics["statistics"]["models"][model]
            total_detections = sum(1 for d in self.metrics["recent_detections"] 
                                 if "models" in d and model in d["models"] and d["models"][model]["detected"])
            
            if self.metrics["recent_detections"]:
                stats["detection_rate"] = total_detections / len(self.metrics["recent_detections"])
                
                confidences = [d["models"][model]["confidence"] for d in self.metrics["recent_detections"] 
                             if "models" in d and model in d["models"]]
                
                if confidences:
                    stats["avg_confidence"] = float(np.mean(confidences))
                    stats["max_confidence"] = float(np.max(confidences))
                    stats["min_confidence"] = float(np.min(confidences))

        # Save updated metrics
        self._save_metrics()
        
        return detection

    def get_metrics(self):
        """Get current metrics"""
        return self.metrics

    def export_to_csv(self, output_file="model_metrics.csv"):
        """Export metrics to CSV for analysis"""
        try:
            if not self.metrics["recent_detections"]:
                logger.warning("No detection data to export.")
                return
                
            # Convert recent detections to DataFrame
            df = pd.DataFrame(self.metrics["recent_detections"])
            
            # Create columns for all models
            all_models = set()
            for detection in self.metrics["recent_detections"]:
                if "models" in detection:
                    all_models.update(detection["models"].keys())
            
            # Flatten nested structures
            for model in all_models:
                df[f'{model}_detected'] = df['models'].apply(
                    lambda x: x.get(model, {}).get('detected', False) if isinstance(x, dict) else False
                )
                df[f'{model}_confidence'] = df['models'].apply(
                    lambda x: x.get(model, {}).get('confidence', 0) if isinstance(x, dict) else 0
                )
            
            # Add agreement columns
            df['agreement_count'] = df['agreement'].apply(lambda x: x.get('count', 0) if isinstance(x, dict) else 0)
            df['agreement_percentage'] = df['agreement'].apply(lambda x: x.get('percentage', 0) if isinstance(x, dict) else 0)
            df['all_agree'] = df['agreement'].apply(lambda x: x.get('all_agree', False) if isinstance(x, dict) else False)
            df['majority_agree'] = df['agreement'].apply(lambda x: x.get('majority_agree', False) if isinstance(x, dict) else False)
            df['any_detected'] = df['agreement'].apply(lambda x: x.get('any_detected', False) if isinstance(x, dict) else False)
            
            # Drop the nested columns
            columns_to_drop = []
            for col in df.columns:
                if col.endswith('_detections') or col in ['models', 'agreement']:
                    columns_to_drop.append(col)
            
            df = df.drop(columns_to_drop, axis=1, errors='ignore')
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Metrics exported to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting metrics to CSV: {e}") 