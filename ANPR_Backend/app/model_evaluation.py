import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch
from torchvision import transforms
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import logging
from datetime import datetime
import json
from weapon_detection import (
    yolo_model_s, yolo_v8_model, yolo_v11_model, 
    pytorch_model, kaggle_model, keras_model,
    preprocess_for_pytorch, preprocess_for_keras,
    process_yolo_results, process_yolov8_results,
    normalize_prediction
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
test_images_dir = os.path.join(base_dir, "test_images")
ground_truth_file = os.path.join(base_dir, "ground_truth.json")
results_dir = os.path.join(base_dir, "evaluation_results")
os.makedirs(results_dir, exist_ok=True)

# Define thresholds for different models
THRESHOLDS = {
    "yolov5s": 0.5,
    "yolov8s": 0.5,
    "yolov11s": 0.5,
    "pytorch": 0.5,
    "kaggle": 0.5,
    "keras": 0.7
}

def load_ground_truth():
    """Load ground truth annotations from JSON file"""
    try:
        with open(ground_truth_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading ground truth: {str(e)}")
        return {}

def evaluate_yolo_model(model, image_path, model_name, threshold):
    """Evaluate a YOLO model on a single image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        if model_name in ["yolov8s", "yolov11s"]:
            results = model(img)
            detections = process_yolov8_results(results, model_name)
        else:
            results = model(img)
            detections = process_yolo_results(results, model_name)
        
        # Filter detections by confidence threshold
        detections = [d for d in detections if d["confidence"] >= threshold]
        
        # Filter out person detections
        detections = [d for d in detections if d["category"] not in ["person"]]
        
        return detections
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {str(e)}")
        return []

def evaluate_pytorch_model(model, image_path, threshold):
    """Evaluate PyTorch model on a single image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        pytorch_input = preprocess_for_pytorch(img)
        with torch.no_grad():
            pytorch_output = model(pytorch_input)
            pytorch_pred = torch.sigmoid(pytorch_output).item()
        
        if pytorch_pred >= threshold:
            return [{
                "category": "weapon",
                "confidence": float(pytorch_pred),
                "model": "pytorch"
            }]
        return []
    except Exception as e:
        logger.error(f"Error evaluating PyTorch model: {str(e)}")
        return []

def evaluate_kaggle_model(model, image_path, threshold):
    """Evaluate Kaggle model on a single image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        kaggle_input = preprocess_for_pytorch(img)
        with torch.no_grad():
            kaggle_output = model(kaggle_input)
            kaggle_probs = torch.softmax(kaggle_output, dim=1)
            kaggle_pred, kaggle_class = torch.max(kaggle_probs, dim=1)
            kaggle_pred = kaggle_pred.item()
            kaggle_class = kaggle_class.item()
        
        weapon_classes = {
            0: 'Automatic Rifle', 1: 'Bazooka', 2: 'Grenade Launcher',
            3: 'Handgun', 4: 'Knife', 5: 'Shotgun',
            6: 'SMG', 7: 'Sniper', 8: 'Sword'
        }
        
        weapon_type = weapon_classes.get(kaggle_class, 'Unknown')
        
        if kaggle_pred >= threshold:
            return [{
                "category": "weapon",
                "specific_type": weapon_type,
                "confidence": float(kaggle_pred),
                "model": "kaggle"
            }]
        return []
    except Exception as e:
        logger.error(f"Error evaluating Kaggle model: {str(e)}")
        return []

def evaluate_keras_model(model, image_path, threshold):
    """Evaluate Keras model on a single image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        keras_input = preprocess_for_keras(img)
        raw_prediction = float(model.predict(keras_input, verbose=0)[0][0])
        normalized_pred = normalize_prediction(raw_prediction)
        
        if normalized_pred >= threshold:
            return [{
                "category": "weapon",
                "confidence": normalized_pred,
                "model": "keras"
            }]
        return []
    except Exception as e:
        logger.error(f"Error evaluating Keras model: {str(e)}")
        return []

def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, F1 score and confusion matrix"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }

def evaluate_models():
    """Evaluate all models on test images and generate CSV report"""
    # Load ground truth
    ground_truth = load_ground_truth()
    
    # Initialize results dictionary
    results = {
        "yolov5s": {"y_true": [], "y_pred": [], "detections": []},
        "yolov8s": {"y_true": [], "y_pred": [], "detections": []},
        "yolov11s": {"y_true": [], "y_pred": [], "detections": []},
        "pytorch": {"y_true": [], "y_pred": [], "detections": []},
        "kaggle": {"y_true": [], "y_pred": [], "detections": []},
        "keras": {"y_true": [], "y_pred": [], "detections": []}
    }
    
    # Process each test image
    for image_name, gt_data in ground_truth.items():
        image_path = os.path.join(test_images_dir, image_name)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
        
        # Get ground truth label
        has_weapon = gt_data.get("has_weapon", False)
        
        # Evaluate YOLOv5s
        if yolo_model_s:
            detections = evaluate_yolo_model(yolo_model_s, image_path, "yolov5s", THRESHOLDS["yolov5s"])
            results["yolov5s"]["y_true"].append(has_weapon)
            results["yolov5s"]["y_pred"].append(len(detections) > 0)
            results["yolov5s"]["detections"].append(detections)
        
        # Evaluate YOLOv8s
        if yolo_v8_model:
            detections = evaluate_yolo_model(yolo_v8_model, image_path, "yolov8s", THRESHOLDS["yolov8s"])
            results["yolov8s"]["y_true"].append(has_weapon)
            results["yolov8s"]["y_pred"].append(len(detections) > 0)
            results["yolov8s"]["detections"].append(detections)
        
        # Evaluate YOLOv11s
        if yolo_v11_model:
            detections = evaluate_yolo_model(yolo_v11_model, image_path, "yolov11s", THRESHOLDS["yolov11s"])
            results["yolov11s"]["y_true"].append(has_weapon)
            results["yolov11s"]["y_pred"].append(len(detections) > 0)
            results["yolov11s"]["detections"].append(detections)
        
        # Evaluate PyTorch model
        if pytorch_model:
            detections = evaluate_pytorch_model(pytorch_model, image_path, THRESHOLDS["pytorch"])
            results["pytorch"]["y_true"].append(has_weapon)
            results["pytorch"]["y_pred"].append(len(detections) > 0)
            results["pytorch"]["detections"].append(detections)
        
        # Evaluate Kaggle model
        if kaggle_model:
            detections = evaluate_kaggle_model(kaggle_model, image_path, THRESHOLDS["kaggle"])
            results["kaggle"]["y_true"].append(has_weapon)
            results["kaggle"]["y_pred"].append(len(detections) > 0)
            results["kaggle"]["detections"].append(detections)
        
        # Evaluate Keras model
        if keras_model:
            detections = evaluate_keras_model(keras_model, image_path, THRESHOLDS["keras"])
            results["keras"]["y_true"].append(has_weapon)
            results["keras"]["y_pred"].append(len(detections) > 0)
            results["keras"]["detections"].append(detections)
    
    # Calculate metrics for each model
    metrics = {}
    for model_name, model_results in results.items():
        if model_results["y_true"]:  # Only calculate if we have results
            metrics[model_name] = calculate_metrics(
                model_results["y_true"],
                model_results["y_pred"]
            )
    
    # Generate detailed CSV report
    csv_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(results_dir, f"model_evaluation_{timestamp}.csv")
    
    for model_name, model_results in results.items():
        if model_name in metrics:
            model_metrics = metrics[model_name]
            for i, (y_true, y_pred, detections) in enumerate(zip(
                model_results["y_true"],
                model_results["y_pred"],
                model_results["detections"]
            )):
                # Get image name from ground truth
                image_name = list(ground_truth.keys())[i]
                
                # Calculate average confidence for detections
                avg_confidence = np.mean([d["confidence"] for d in detections]) if detections else 0
                
                # Get specific weapon types for Kaggle model
                weapon_types = []
                if model_name == "kaggle" and detections:
                    weapon_types = [d.get("specific_type", "Unknown") for d in detections]
                
                csv_data.append({
                    "Model": model_name,
                    "Image": image_name,
                    "Ground_Truth": y_true,
                    "Prediction": y_pred,
                    "Correct": y_true == y_pred,
                    "Detections_Count": len(detections),
                    "Average_Confidence": avg_confidence,
                    "Weapon_Types": ", ".join(weapon_types) if weapon_types else "N/A",
                    "Precision": model_metrics["precision"],
                    "Recall": model_metrics["recall"],
                    "F1_Score": model_metrics["f1"]
                })
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    logger.info(f"Evaluation results saved to {csv_file}")
    
    # Save detailed metrics to JSON
    metrics_file = os.path.join(results_dir, f"model_metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Detailed metrics saved to {metrics_file}")
    
    return metrics, csv_file

if __name__ == "__main__":
    metrics, csv_file = evaluate_models()
    print(f"\nEvaluation complete! Results saved to: {csv_file}")
    print("\nModel Metrics Summary:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.upper()}:")
        print(f"Precision: {model_metrics['precision']:.3f}")
        print(f"Recall: {model_metrics['recall']:.3f}")
        print(f"F1 Score: {model_metrics['f1']:.3f}")