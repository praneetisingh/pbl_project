# app/main.py
import os
import shutil
import cv2
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.anpr import recognize_license_plate_from_image, recognize_license_plate_from_video, recognize_license_plate_from_frame
import asyncio
from fastapi import WebSocketDisconnect
from app.weapon_detection import detect_weapons_from_image
from .model_metrics import ModelMetricsTracker
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Weapon Detection API",
    description="API for uploading images to detect weapons.",
    version="1.0.0"
)

# Initialize metrics tracker
metrics_tracker = ModelMetricsTracker()

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:5500",  # Added frontend origin
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Update as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded files temporarily
UPLOAD_DIR = "uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to handle image uploads for weapon detection.
    """
    filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    logger.info(f"Received file upload: {filename}")

    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logger.info(f"File saved temporarily at: {file_path}")

        # Process only image files
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in [".jpg", ".jpeg", ".png"]:
            try:
                logger.info("Starting weapon detection...")
                results = detect_weapons_from_image(file_path)
                logger.info("Weapon detection completed successfully")

                # Add to metrics
                metrics_tracker.add_detection(
                    image_name=filename,
                    detection_results=results
                )

                return JSONResponse(content={
                    "filename": filename,
                    "results": results,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error during weapon detection: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": str(e),
                        "message": "Error during weapon detection",
                        "filename": filename
                    }
                )
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Unsupported file type",
                    "message": "Please upload an image file (jpg, jpeg, or png)",
                    "filename": filename,
                    "file_type": file_extension
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "An unexpected error occurred",
                "filename": filename
            }
        )
    finally:
        # Clean up the uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file removed: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary file {file_path}: {str(e)}")

@app.websocket("/live/")
async def live_camera(websocket: WebSocket):
    """
    WebSocket endpoint to stream live camera feed and send detected license plate details.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    # Initialize camera with a higher resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        logger.error("Could not access the camera")
        await websocket.send_json({"error": "Could not access the camera."})
        await websocket.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read from camera")
                await websocket.send_json({"error": "Failed to read from camera."})
                break

            # Process the frame to detect license plates
            results = recognize_license_plate_from_frame(frame)
            
            # Always send a response, whether plates are detected or not
            if results:
                await websocket.send_json({"status": "success", "results": results})
            else:
                await websocket.send_json({"status": "scanning", "message": "No plates detected"})
            
            # Add a small delay to prevent overwhelming the connection
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket: {str(e)}")
        await websocket.send_json({"error": str(e)})
    finally:
        cap.release()
        await websocket.close()

@app.get("/metrics")
async def get_metrics():
    """Get current model metrics"""
    try:
        metrics = metrics_tracker.get_metrics()
        # Force reload from file to ensure latest data
        updated_metrics = ModelMetricsTracker().get_metrics()
        return JSONResponse(content=updated_metrics)
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export-metrics")
async def export_metrics(output_file: str = "weapon_detection_metrics.csv"):
    """Export metrics to CSV file"""
    try:
        # Create a full path for the output file
        if not os.path.isabs(output_file):
            output_file = os.path.join(os.getcwd(), output_file)
            
        metrics_tracker.export_to_csv(output_file)
        return JSONResponse(content={"status": "success", "file": output_file})
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)