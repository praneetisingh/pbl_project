import os
import torch
import requests
import sys
from pathlib import Path

def download_file(url, dest_path):
    """Download a file from a URL to a destination path."""
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Successfully downloaded to {dest_path}")

def main():
    # Create models directory if it doesn't exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Download YOLOv9
    try:
        yolov9_path = os.path.join(models_dir, 'yolov9s.pt')
        if not os.path.exists(yolov9_path):
            download_file(
                "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt",
                yolov9_path
            )
        print("Now you can use YOLOv9 by updating the path in weapon_detection.py:")
        print(f"yolov9_model_path = '{yolov9_path}'")
        print("yolo_v9_model = torch.hub.load('WongKinYiu/yolov9', 'custom', path=yolov9_model_path)")
    except Exception as e:
        print(f"Error downloading YOLOv9: {str(e)}")
    
    # YOLOv10 (currently not available from Ultralytics)
    try:
        # Use YOLOv8 instead for compatibility
        print("YOLOv10 is not officially available yet. Using YOLOv8 as a substitute.")
        yolov10_path = os.path.join(models_dir, 'yolov10s.pt')
        if not os.path.exists(yolov10_path):
            # Currently using YOLOv8 for v10 since it's not officially released
            from ultralytics import YOLO
            model = YOLO("yolov8s.pt")
            model.export(format="pt", save=True)
            # Copy the exported file to the v10 path for compatibility
            if os.path.exists("yolov8s.pt"):
                import shutil
                shutil.copy("yolov8s.pt", yolov10_path)
                print(f"Created placeholder for YOLOv10 at {yolov10_path}")
    except Exception as e:
        print(f"Error setting up YOLOv10: {str(e)}")
    
    # YOLOv11 (currently not available from Ultralytics)
    try:
        # Use YOLOv8 instead for compatibility
        print("YOLOv11 is not officially available yet. Using YOLOv8 as a substitute.")
        yolov11_path = os.path.join(models_dir, 'yolov11s.pt')
        if not os.path.exists(yolov11_path):
            # Currently using YOLOv8 for v11 since it's not officially released
            from ultralytics import YOLO
            model = YOLO("yolov8s.pt")
            # Copy the exported file to the v11 path for compatibility
            if os.path.exists("yolov8s.pt"):
                import shutil
                shutil.copy("yolov8s.pt", yolov11_path)
                print(f"Created placeholder for YOLOv11 at {yolov11_path}")
    except Exception as e:
        print(f"Error setting up YOLOv11: {str(e)}")
    
    # YOLOv12 from sunsmarterjie
    try:
        yolov12_path = os.path.join(models_dir, 'yolov12s.pt')
        if not os.path.exists(yolov12_path):
            download_file(
                "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0.0/yolov12-s.pt",
                yolov12_path
            )
        print("Now you can use YOLOv12 by updating the path in weapon_detection.py:")
        print(f"yolov12_model_path = '{yolov12_path}'")
        print("yolo_v12_model = torch.load(yolov12_model_path)")
    except Exception as e:
        print(f"Error downloading YOLOv12: {str(e)}")
        
    print("\nUpdate weapon_detection.py to use local model files instead of torch.hub.load")

if __name__ == "__main__":
    main() 