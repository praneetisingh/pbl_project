import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import requests
from PIL import Image, ImageTk
import os
import threading
import json
import websocket
from datetime import datetime
import cv2
import numpy as np

# Update these URLs if needed
API_URL = "http://127.0.0.1:8000/upload/"
WEBSOCKET_URL = "ws://127.0.0.1:8000/live/"

class WeaponDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Weapon Detection System")
        self.root.geometry("1200x800")
        
        # Create main container
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tab_control = ttk.Notebook(self.main_container)
        self.detection_tab = ttk.Frame(self.tab_control)
        self.logs_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.detection_tab, text='Detection')
        self.tab_control.add(self.logs_tab, text='Logs')
        self.tab_control.pack(expand=True, fill=tk.BOTH)
        
        self.setup_detection_tab()
        self.setup_logs_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Available models tracking
        self.available_models = []

    def setup_detection_tab(self):
        # Left frame for controls
        left_frame = ttk.Frame(self.detection_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Upload button
        upload_label = ttk.Label(left_frame, text="Select an image to detect weapons:")
        upload_label.pack(pady=5)
        
        self.upload_btn = ttk.Button(left_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=5)
        
        # Refresh button
        refresh_label = ttk.Label(left_frame, text="Force refresh results:")
        refresh_label.pack(pady=(20, 5))
        
        self.refresh_btn = ttk.Button(left_frame, text="Refresh Data", command=self.refresh_data)
        self.refresh_btn.pack(pady=5)
        
        # Export metrics button
        self.export_btn = ttk.Button(left_frame, text="Export Results", command=self.export_metrics)
        self.export_btn.pack(pady=20)
        
        # Right frame for results with scrollbar
        right_frame = ttk.Frame(self.detection_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a canvas and scrollbar for the results
        self.canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Results display
        results_label = ttk.Label(self.scrollable_frame, text="Detection Results:", font=("Arial", 12, "bold"))
        results_label.pack(anchor="w", pady=(0, 5))
        
        self.results_text = scrolledtext.ScrolledText(self.scrollable_frame, height=10, width=80, font=("Arial", 10))
        self.results_text.pack(fill=tk.X, expand=True)
        
        # Image display
        image_label = ttk.Label(self.scrollable_frame, text="Analyzed Image:", font=("Arial", 12, "bold"))
        image_label.pack(anchor="w", pady=(20, 5))
        
        self.image_frame = ttk.Frame(self.scrollable_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(pady=5)

    def setup_logs_tab(self):
        # Create log display area
        log_frame = ttk.Frame(self.logs_tab)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add controls for log filtering
        filter_frame = ttk.Frame(log_frame)
        filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(filter_frame, text="Filter logs:").pack(side=tk.LEFT, padx=5)
        self.log_filter = ttk.Entry(filter_frame, width=30)
        self.log_filter.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(filter_frame, text="Apply Filter", command=self.filter_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Clear Filter", command=self.clear_log_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Refresh Logs", command=self.refresh_logs).pack(side=tk.LEFT, padx=5)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(log_frame, height=30)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Load logs initially
        self.refresh_logs()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.status_var.set("Processing image...")
            threading.Thread(target=self.process_image, args=(file_path,)).start()

    def process_image(self, file_path):
        try:
            # Upload image to backend
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    results = response.json()
                    self.display_results(results)
                    self.display_image(file_path)
                    self.status_var.set("Processing complete")
                else:
                    self.status_var.set("Error processing image")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")

    def display_image(self, file_path):
        """Display the processed image with detection overlays"""
        try:
            # Load the image
            img = Image.open(file_path)
            
            # Resize image to fit in the window
            width, height = img.size
            max_width = 800
            if width > max_width:
                ratio = max_width / width
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage for tkinter
            img_tk = ImageTk.PhotoImage(img)
            
            # Update the label
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk  # Keep a reference
        except Exception as e:
            print(f"Error displaying image: {e}")
   
    def display_results(self, results):
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Display filename
        self.results_text.insert(tk.END, f"File: {results['filename']}\n\n")

        # Show final detection result prominently
        weapon_detected = results.get("weapon_detected", False)
        self.results_text.insert(tk.END, f"FINAL RESULT: {'⚠️ WEAPON DETECTED!' if weapon_detected else 'No weapons detected'}\n\n")
        
        # Display agreement statistics
        agreement = results.get("agreement", {})
        self.results_text.insert(tk.END, f"Model Agreement: {agreement.get('percentage', 0):.1f}% ({agreement.get('count', 0)} out of {agreement.get('total_models', 0)} models)\n\n")
        
        # Display minimal model results
        yolo_models = [
            ("YOLOv5s", "yolov5s_detections"),
            ("YOLOv8s", "yolov8s_detections"),
            ("YOLOv11s", "yolov11s_detections")
        ]
        
        self.results_text.insert(tk.END, "Individual Model Results:\n")
        
        # Display simple model results
        stats = results.get("model_statistics", {})
        for model, model_stats in stats.items():
            detected = model_stats.get('detected', False)
            confidence = model_stats.get('confidence', 0)
            self.results_text.insert(tk.END, f"- {model.upper()}: {'Detected' if detected else 'Not Detected'} (Confidence: {confidence:.2%})\n")
        
        # Update status bar based on detection results
        if weapon_detected:
            self.status_var.set("⚠️ WEAPON DETECTED!")
        else:
            self.status_var.set("No weapons detected")

    def refresh_data(self):
        """Force refresh all data from the backend"""
        self.status_var.set("Refreshing data...")
        try:
            # Get the base URL from API_URL
            base_url = API_URL.rsplit('/', 2)[0]
            
            # Clear the logs first to avoid confusion
            self.refresh_logs()
            
            # Fetch the latest metrics data
            response = requests.get(f"{base_url}/metrics")
            if response.status_code == 200:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Data refreshed successfully!\n\n")
                
                metrics = response.json()
                if "recent_detections" in metrics and metrics["recent_detections"]:
                    latest = metrics["recent_detections"][0]
                    self.results_text.insert(tk.END, f"Latest detection from: {latest.get('image_name', 'Unknown')}\n")
                    self.results_text.insert(tk.END, f"Time: {latest.get('timestamp', 'Unknown')[:19]}\n\n")
                    
                    # Show model results
                    self.results_text.insert(tk.END, "Model Results:\n")
                    if "models" in latest:
                        for model, data in latest["models"].items():
                            detected = data.get("detected", False)
                            confidence = data.get("confidence", 0)
                            self.results_text.insert(tk.END, f"- {model.upper()}: {'Detected' if detected else 'Not Detected'} (Confidence: {confidence:.2%})\n")
                    
                    # Show agreement
                    if "agreement" in latest:
                        agreement = latest["agreement"]
                        self.results_text.insert(tk.END, f"\nAgreement: {agreement.get('percentage', 0):.1f}% ({agreement.get('count', 0)} out of {agreement.get('total_models', 0)} models)\n")
                else:
                    self.results_text.insert(tk.END, "No recent detections found.\n")
                
                self.status_var.set("Data refreshed successfully")
            else:
                self.status_var.set(f"Error refreshing data: {response.status_code}")
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Error refreshing data: {response.status_code}\n")
                self.results_text.insert(tk.END, response.text)
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error refreshing data: {str(e)}\n")

    def refresh_results(self):
        """Refresh the logs"""
        self.refresh_logs()

    def export_metrics(self):
        """Export metrics to CSV file"""
        try:
            # Ask for a filename
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Metrics"
            )
            
            if file_path:
                # Use the base API URL without the 'upload/' part
                base_url = API_URL.rsplit('/', 2)[0]
                export_url = f"{base_url}/export-metrics"
                
                # Make a request to export the metrics
                response = requests.get(export_url, params={"output_file": file_path})
                
                if response.status_code == 200:
                    self.status_var.set(f"Metrics exported to {file_path}")
                else:
                    self.status_var.set("Error exporting metrics")
        except Exception as e:
            self.status_var.set(f"Error exporting metrics: {e}")

    def filter_logs(self):
        """Filter logs based on the filter text"""
        filter_text = self.log_filter.get().lower()
        self.refresh_logs(filter_text)

    def clear_log_filter(self):
        """Clear the log filter and refresh logs"""
        self.log_filter.delete(0, tk.END)
        self.refresh_logs()

    def refresh_logs(self, filter_text=None):
        """Refresh and load logs from the log file"""
        try:
            # Path to the detection log file
            log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'detection_logs.json')
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                
                # Clear previous logs
                self.log_text.delete(1.0, tk.END)
                
                # Display logs in reverse order (newest first)
                for log in reversed(logs):
                    log_text = json.dumps(log, indent=2)
                    
                    # Apply filter if provided
                    if filter_text and filter_text not in log_text.lower():
                        continue
                    
                    self.log_text.insert(tk.END, log_text + "\n\n" + "-" * 80 + "\n\n")
            else:
                self.log_text.delete(1.0, tk.END)
                self.log_text.insert(tk.END, "No log file found.")
        except Exception as e:
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, f"Error loading logs: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WeaponDetectorGUI(root)
    root.mainloop()