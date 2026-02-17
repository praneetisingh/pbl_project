import torch
import cv2
import numpy as np
from torchvision import transforms
import tensorflow as tf
from tensorflow import keras
import os
import logging
from PIL import Image
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the model files
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

# Define model paths
yolo_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov5s.pt")
yolov8_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8s.pt")
yolov11_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov11s.pt")
pytorch_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_weapon_detector_custom.pth")
kaggle_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_weapon_detector_kaggle.pth")
keras_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weapon_detection_final.keras")

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), exist_ok=True)

# Initialize models
yolo_model_s = None
yolo_v8_model = None
yolo_v11_model = None
pytorch_model = None
kaggle_model = None
keras_model = None

# Load YOLOv5s model
try:
    yolo_model_s = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=True)
    yolo_model_s.eval()
    logger.info("YOLOv5s model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLOv5s model: {str(e)}")
    yolo_model_s = None

# Load YOLOv8s model
try:
    from ultralytics import YOLO
    yolo_v8_model = YOLO(yolov8_model_path)
    logger.info("YOLOv8s model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLOv8s model: {str(e)}")
    yolo_v8_model = None

# Load YOLOv11 model
try:
    # Try loading YOLOv11 using YOLO class instead of torch.hub
    try:
        from ultralytics import YOLO
        yolo_v11_model = YOLO(yolov11_model_path)
        logger.info("YOLOv11 model loaded successfully with YOLO class")
    except Exception as e1:
        # Fallback to torch.hub with a different source
        logger.warning(f"Failed to load YOLOv11 with YOLO class: {str(e1)}")
        yolo_v11_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov11_model_path, force_reload=True, trust_repo=True)
        yolo_v11_model.eval()
        logger.info("YOLOv11 model loaded successfully with torch.hub")
except Exception as e:
    logger.error(f"Error loading YOLOv11 model: {str(e)}")
    yolo_v11_model = None

# Load PyTorch model
try:
    if os.path.exists(pytorch_model_path):
        # Use map_location to load models saved on CUDA devices onto CPU
        state_dict = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
        
        # Create a custom CNN model that matches the structure in the saved state dict
        class CustomCNN(torch.nn.Module):
            def __init__(self):
                super(CustomCNN, self).__init__()
                # Based on the error message, update the dimensions to match the saved model
                self.conv1 = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2)
                )
                self.conv2 = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2)
                )
                self.conv3 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2)
                )
                self.conv4 = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2)
                )
                # Fully connected layers
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(256 * 14 * 14, 512),  # 50176 = 256 * 14 * 14
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(512, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(128, 1)
                )
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = x.view(x.size(0), -1)  # Flatten
                x = self.fc(x)
                return x
        
        # Create the model and load the state dict
        pytorch_model = CustomCNN()
        
        # Add debugging before loading state_dict
        logger.info(f"Model created with structure matching the saved state dict")
        
        # Try to load the state dict
        try:
            pytorch_model.load_state_dict(state_dict)
            logger.info("State dict loaded successfully")
        except Exception as load_error:
            logger.error(f"Error loading state dict: {str(load_error)}")
            # Continue with the model even if state dict loading failed
            
        pytorch_model.eval()
        logger.info("Custom PyTorch CNN model ready for inference")
    else:
        logger.warning(f"PyTorch model file not found at {pytorch_model_path}")
        pytorch_model = None
except Exception as e:
    logger.error(f"Error loading PyTorch model: {str(e)}")
    pytorch_model = None

# Load Kaggle-trained PyTorch model
try:
    if os.path.exists(kaggle_model_path):
        # Use map_location to load models saved on CUDA devices onto CPU
        state_dict = torch.load(kaggle_model_path, map_location=torch.device('cpu'))
        
        # Create a WeaponDetector model that matches the structure in the saved state dict
        class WeaponDetector(torch.nn.Module):
            def __init__(self, num_classes=9):
                super(WeaponDetector, self).__init__()
                
                # Initial convolution
                self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(64)
                self.relu = torch.nn.ReLU(inplace=True)
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                
                # Residual blocks
                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                self.layer4 = self._make_layer(256, 512, 2, stride=2)
                
                # Global average pooling and classifier
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(512, 512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(512, num_classes)
                )
            
            def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
                layers = []
                layers.append(ResidualBlock(in_channels, out_channels, stride))
                for _ in range(1, num_blocks):
                    layers.append(ResidualBlock(out_channels, out_channels))
                return torch.nn.Sequential(*layers)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                
                return x
        
        class ResidualBlock(torch.nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super(ResidualBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                          stride=stride, padding=1, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(out_channels)
                self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                          stride=1, padding=1, bias=False)
                self.bn2 = torch.nn.BatchNorm2d(out_channels)
                
                self.shortcut = torch.nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                     stride=stride, bias=False),
                        torch.nn.BatchNorm2d(out_channels)
                    )
            
            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = torch.relu(out)
                return out
        
        # Create the model and load the state dict
        kaggle_model = WeaponDetector()
        
        # Add debugging before loading state_dict
        logger.info(f"Kaggle model created with structure matching the saved state dict")
        
        # Try to load the state dict
        try:
            kaggle_model.load_state_dict(state_dict)
            logger.info("Kaggle model state dict loaded successfully")
        except Exception as load_error:
            logger.error(f"Error loading Kaggle model state dict: {str(load_error)}")
            # Continue with the model even if state dict loading failed
            
        kaggle_model.eval()
        logger.info("Kaggle-trained PyTorch model ready for inference")
    else:
        logger.warning(f"Kaggle model file not found at {kaggle_model_path}")
        kaggle_model = None
except Exception as e:
    logger.error(f"Error loading Kaggle model: {str(e)}")
    kaggle_model = None

# Load Keras model
try:
    if os.path.exists(keras_model_path):
        keras_model = keras.models.load_model(keras_model_path)
        logger.info("Keras model loaded successfully")
    else:
        logger.warning(f"Keras model file not found at {keras_model_path}")
        keras_model = None
except Exception as e:
    logger.error(f"Error loading Keras model: {str(e)}")
    keras_model = None

# Log file for detection results
detection_log_file = os.path.join(base_dir, 'detection_logs.json')

# Define transforms for PyTorch model
pytorch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define weapon classes
WEAPON_CLASSES = {
    'knife': 'knife',
    'gun': 'gun',
    'pistol': 'gun',
    'rifle': 'gun',
    'shotgun': 'gun',
    'weapon': 'other',
    'person': 'person'
}

def preprocess_for_pytorch(image):
    """Preprocess image for PyTorch model input"""
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = pytorch_transform(image)
        image = image.unsqueeze(0)
        return image
    except Exception as e:
        logger.error(f"Error in PyTorch preprocessing: {str(e)}")
        raise

def preprocess_for_keras(image):
    """Preprocess image for Keras model input"""
    try:
        resized = cv2.resize(image, (224, 224))
        preprocessed = np.expand_dims(resized, axis=0) / 255.0
        return preprocessed
    except Exception as e:
        logger.error(f"Error in Keras preprocessing: {str(e)}")
        raise

def normalize_prediction(prediction):
    """Normalize the prediction to a more reasonable range"""
    try:
        pred = float(prediction)
        if pred > 0.99:
            logger.warning(f"Extremely high prediction detected: {pred}")
            pred = 0.5 + (pred - 0.5) * 0.5
        pred = max(0.0, min(1.0, pred))
        return pred
    except Exception as e:
        logger.error(f"Error in prediction normalization: {str(e)}")
        return 0.5

def process_yolo_results(yolo_results, model_name):
    """Process YOLO model results into a standard format"""
    detections = []
    for det in yolo_results.xyxy[0]:
        x1, y1, x2, y2, conf, cls_idx = det.cpu().numpy()
        cls_name = yolo_results.names[int(cls_idx)]
        detections.append({
            "category": cls_name,
            "specific_type": cls_name,
            "confidence": float(conf),
            "coordinates": {
                "x": float(x1),
                "y": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1)
            },
            "model": model_name
        })
    return detections

def process_yolov8_results(results, model_name):
    """Process YOLOv8 results into a standard format"""
    detections = []
    
    if results and len(results) > 0:
        for result in results:
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_idx = int(box.cls[0].cpu().numpy())
                    cls_name = result.names[cls_idx]
                    
                    detections.append({
                        "category": cls_name,
                        "specific_type": cls_name,
                        "confidence": conf,
                        "coordinates": {
                            "x": float(x1),
                            "y": float(y1),
                            "width": float(x2 - x1),
                            "height": float(y2 - y1)
                        },
                        "model": model_name
                    })
    
    return detections

def log_detection_results(results):
    """Log detection results to JSON file"""
    try:
        # Add timestamp
        results['timestamp'] = datetime.now().isoformat()
        
        # Load existing log if it exists
        if os.path.exists(detection_log_file):
            with open(detection_log_file, 'r') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []
        
        # Add new results
        logs.append(results)
        
        # Keep only last 1000 results
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Save logs
        with open(detection_log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error logging detection results: {str(e)}")

def detect_weapons_from_image(image_path):
    """Detect weapons in an image using multiple models"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # YOLOv5s detection (small)
        yolo_s_results = yolo_model_s(img)
        yolo_s_detections = process_yolo_results(yolo_s_results, "YOLOv5s")
        
        # YOLOv8 detection
        yolo_v8_detections = []
        if yolo_v8_model:
            try:
                yolo_v8_results = yolo_v8_model(img)
                yolo_v8_detections = process_yolov8_results(yolo_v8_results, "YOLOv8s")
            except Exception as e:
                logger.error(f"Error in YOLOv8 detection: {str(e)}")
        
        # YOLOv11 detection
        yolo_v11_detections = []
        if yolo_v11_model:
            try:
                # Check which type of model it is
                if hasattr(yolo_v11_model, 'predict'):  # YOLO class
                    yolo_v11_results = yolo_v11_model(img)
                    yolo_v11_detections = process_yolov8_results(yolo_v11_results, "YOLOv11s")
                else:  # torch.hub model
                    yolo_v11_results = yolo_v11_model(img)
                    yolo_v11_detections = process_yolo_results(yolo_v11_results, "YOLOv11s")
            except Exception as e:
                logger.error(f"Error in YOLOv11 detection: {str(e)}")
        
        # PyTorch detection
        pytorch_detections = []
        if pytorch_model is not None:
            try:
                pytorch_input = preprocess_for_pytorch(img)
                with torch.no_grad():
                    pytorch_output = pytorch_model(pytorch_input)
                    pytorch_pred = torch.sigmoid(pytorch_output).item()
                
                if pytorch_pred > 0.5:
                    pytorch_detections.append({
                        "category": "weapon",
                        "confidence": float(pytorch_pred),
                        "coordinates": {
                            "x": 0,
                            "y": 0,
                            "width": width,
                            "height": height
                        },
                        "model": "PyTorch"
                    })
            except Exception as e:
                logger.error(f"Error in PyTorch detection: {str(e)}")
        
        # Kaggle model detection
        kaggle_detections = []
        if kaggle_model is not None:
            try:
                kaggle_input = preprocess_for_pytorch(img)
                with torch.no_grad():
                    kaggle_output = kaggle_model(kaggle_input)
                    # Get the class with highest probability
                    kaggle_probs = torch.softmax(kaggle_output, dim=1)
                    kaggle_pred, kaggle_class = torch.max(kaggle_probs, dim=1)
                    kaggle_pred = kaggle_pred.item()
                    kaggle_class = kaggle_class.item()
                
                # Map class index to weapon type
                weapon_classes = {
                    0: 'Automatic Rifle', 1: 'Bazooka', 2: 'Grenade Launcher',
                    3: 'Handgun', 4: 'Knife', 5: 'Shotgun',
                    6: 'SMG', 7: 'Sniper', 8: 'Sword'
                }
                
                weapon_type = weapon_classes.get(kaggle_class, 'Unknown')
                
                if kaggle_pred > 0.5:
                    kaggle_detections.append({
                        "category": "weapon",
                        "specific_type": weapon_type,
                        "confidence": float(kaggle_pred),
                        "coordinates": {
                            "x": 0,
                            "y": 0,
                            "width": width,
                            "height": height
                        },
                        "model": "Kaggle"
                    })
            except Exception as e:
                logger.error(f"Error in Kaggle model detection: {str(e)}")
        
        # Keras detection
        keras_detections = []
        if keras_model is not None:
            try:
                keras_input = preprocess_for_keras(img)
                raw_prediction = float(keras_model.predict(keras_input, verbose=0)[0][0])
                normalized_pred = normalize_prediction(raw_prediction)
                
                if normalized_pred > 0.7:  # Increased threshold from 0.5 to 0.8
                    keras_detections.append({
                        "category": "weapon",
                        "confidence": normalized_pred,
                        "coordinates": {
                            "x": 0,
                            "y": 0,
                            "width": width,
                            "height": height
                        },
                        "model": "Keras"
                    })
            except Exception as e:
                logger.error(f"Error in Keras detection: {str(e)}")
        
        # Count models that are available and working
        active_models_count = 0
        if yolo_model_s: active_models_count += 1
        if yolo_v8_model: active_models_count += 1
        if yolo_v11_model: active_models_count += 1
        if pytorch_model: active_models_count += 1
        if kaggle_model: active_models_count += 1
        if keras_model: active_models_count += 1
        
        # Calculate model agreement (for all available models)
        all_detections = [
            len(yolo_s_detections) > 0 if yolo_model_s else False,
            len(yolo_v8_detections) > 0 if yolo_v8_model else False,
            len(yolo_v11_detections) > 0 if yolo_v11_model else False,
            len(pytorch_detections) > 0 if pytorch_model else False,
            len(kaggle_detections) > 0 if kaggle_model else False,
            len(keras_detections) > 0 if keras_model else False
        ]
        
        # Count only active models
        active_detections = [d for i, d in enumerate(all_detections) if i < len(all_detections)]
        
        agreement_count = sum(active_detections)
        agreement_percentage = (agreement_count / active_models_count) * 100 if active_models_count > 0 else 0
        
        # Filter weapon detections (remove person detections)
        yolo_s_weapon_detections = [d for d in yolo_s_detections if d["category"] not in ["person"]]
        yolo_v8_weapon_detections = [d for d in yolo_v8_detections if d["category"] not in ["person"]]
        yolo_v11_weapon_detections = [d for d in yolo_v11_detections if d["category"] not in ["person"]]
        
        # Prepare model statistics
        model_statistics = {}
        
        if yolo_model_s:
            model_statistics["yolov5s"] = {
                "detected": len(yolo_s_weapon_detections) > 0,
                "confidence": max([d["confidence"] for d in yolo_s_weapon_detections]) if yolo_s_weapon_detections else 0,
                "raw_predictions": [d["confidence"] for d in yolo_s_weapon_detections],
                "detections_count": len(yolo_s_weapon_detections)
            }
        
        if pytorch_model:
            model_statistics["pytorch"] = {
                "detected": len(pytorch_detections) > 0,
                "confidence": pytorch_pred if 'pytorch_pred' in locals() else 0,
                "raw_prediction": pytorch_pred if 'pytorch_pred' in locals() else 0
            }
        
        if kaggle_model:
            model_statistics["kaggle"] = {
                "detected": len(kaggle_detections) > 0,
                "confidence": kaggle_pred if 'kaggle_pred' in locals() else 0,
                "raw_prediction": kaggle_pred if 'kaggle_pred' in locals() else 0,
                "weapon_type": weapon_type if 'weapon_type' in locals() else "Unknown"
            }
        
        if keras_model:
            model_statistics["keras"] = {
                "detected": len(keras_detections) > 0,
                "confidence": normalized_pred if 'normalized_pred' in locals() else 0,
                "raw_prediction": raw_prediction if 'raw_prediction' in locals() else 0
            }
        
        # Add newer YOLO models if available
        if yolo_v8_model:
            model_statistics["yolov8s"] = {
                "detected": len(yolo_v8_weapon_detections) > 0,
                "confidence": max([d["confidence"] for d in yolo_v8_weapon_detections]) if yolo_v8_weapon_detections else 0,
                "raw_predictions": [d["confidence"] for d in yolo_v8_weapon_detections],
                "detections_count": len(yolo_v8_weapon_detections)
            }
            
        if yolo_v11_model:
            model_statistics["yolov11s"] = {
                "detected": len(yolo_v11_weapon_detections) > 0,
                "confidence": max([d["confidence"] for d in yolo_v11_weapon_detections]) if yolo_v11_weapon_detections else 0,
                "raw_predictions": [d["confidence"] for d in yolo_v11_weapon_detections],
                "detections_count": len(yolo_v11_weapon_detections)
            }
        
        # Prepare agreement statistics
        agreement = {
            "count": agreement_count,
            "total_models": active_models_count,
            "percentage": agreement_percentage,
            "all_agree": agreement_count == active_models_count or agreement_count == 0,
            "any_detected": any(active_detections),
            "majority_agree": agreement_count > (active_models_count / 2) if active_models_count > 0 else False
        }
        
        # Prepare final results
        results = {
            "filename": os.path.basename(image_path),
            "model_statistics": model_statistics,
            "agreement": agreement,
            "weapon_detected": any([
                len(yolo_s_weapon_detections) > 0,
                len(yolo_v8_weapon_detections) > 0,
                len(yolo_v11_weapon_detections) > 0,
                len(pytorch_detections) > 0,
                len(kaggle_detections) > 0,
                len(keras_detections) > 0
            ])
        }
        
        # Add model detections if available
        if yolo_model_s:
            results["yolov5s_detections"] = yolo_s_weapon_detections
            
        if yolo_v8_model:
            results["yolov8s_detections"] = yolo_v8_weapon_detections
            
        if yolo_v11_model:
            results["yolov11s_detections"] = yolo_v11_weapon_detections
            
        if pytorch_model:
            results["pytorch_detections"] = pytorch_detections
            
        if kaggle_model:
            results["kaggle_detections"] = kaggle_detections
            
        if keras_model:
            results["keras_detections"] = keras_detections
        
        # Log results to JSON file
        log_detection_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in weapon detection: {str(e)}")
        raise