from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import logging
import sys
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import easyocr
import os
import googlemaps
import re
from googletrans import Translator
from deepface import DeepFace
from twilio.rest import Client

# ================= SETUP ==========================

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

origins = [
    "https://6000-firebase-studio-1746954444859.cluster-fdkw7vjj7bgguspe3fbbc25tra.cloudworkstations.dev",
    "http://localhost:9002",
    "http://127.0.0.1:9002",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= GLOBAL VARIABLES ==========================

YOLO_MODEL_PATH = 'yolov8n.pt'
model = YOLO(YOLO_MODEL_PATH)
reader = easyocr.Reader(['en', 'hi'])
translator = Translator()
mode = "outdoor"
current_lang = "en"

mode_classes = {
    "indoor": ["person", "chair", "sofa", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "book", "cup", "fork", "knife", "spoon", "bowl", "bottle", "wine glass", "umbrella", "handbag", "tie", "suitcase", "pottedplant"],
    "outdoor": ["person", "bicycle", "motorbike", "car", "bus", "train", "truck", "boat", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "kite", "frisbee", "skateboard", "sports ball", "tennis racket", "baseball bat", "baseball glove", "surfboard", "snowboard", "skis"],
    "road": ["person", "car", "motorbike", "bus", "truck", "bicycle", "train", "traffic light", "stop sign", "parking meter", "bench", "fire hydrant", "road sign"]
}

# Google Maps
GOOGLE_MAPS_API_KEY = "AIzaSyAF37kDyerCWtnvCFxz8AqVsv_U_rRmBLI"
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# SOS Credentials

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
USER_PHONE_NUMBER = os.getenv("USER_PHONE_NUMBER")


# Face Recognition Setup
REFERENCE_IMAGE_PATH = r"C:\\AFVP\\WIN_20250527_16_28_39_Pro.jpg"
REFERENCE_NAME = "Sail Nagale"
reference_img = cv2.imread(REFERENCE_IMAGE_PATH)
reference_img_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

# ================= SCHEMAS ==========================

class ImageData(BaseModel):
    image_base64: str

class ModeData(BaseModel):
    mode: str

class LanguageRequest(BaseModel):
    language: str

class GPSRequest(BaseModel):
    destination: str

class FaceData(BaseModel):
    image_base64: str

class SOSRequest(BaseModel):
    location: str

# ================= UTILITIES ==========================

def speak(text, lang=None):
    global current_lang
    if lang is None:
        lang = current_lang
    try:
        if lang != 'en':
            translated = translator.translate(text, dest=lang)
            text = translated.text
        tts = gTTS(text=text, lang=lang)
        tts.save("speech.mp3")
        sound = AudioSegment.from_mp3("speech.mp3")
        play(sound)
        os.remove("speech.mp3")
    except Exception as e:
        logger.error(f"Speech error: {e}")

def base64_to_cv2_image(data_uri):
    try:
        header, encoded = data_uri.split(",", 1)
        decoded_data = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Base64 decode error: {e}")
        return None

# ================= ROUTES ==========================

@app.get("/")
async def root():
    return {"message": "FastAPI server is running!"}

@app.post("/set_mode")
async def set_mode(data: ModeData):
    global mode
    if data.mode.lower() not in mode_classes:
        raise HTTPException(status_code=400, detail="Invalid mode.")
    mode = data.mode.lower()
    return {"message": f"Mode switched to {mode}"}

@app.post("/set_language")
async def set_language(data: LanguageRequest):
    global current_lang
    supported_languages = ['en', 'hi', 'mr', 'ta', 'te']
    if data.language not in supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")
    current_lang = data.language
    return {"message": f"Language set to {current_lang}"}

@app.post("/start_detection")
async def start_detection(image_data: ImageData):
    try:
        cv2_img = base64_to_cv2_image(image_data.image_base64)
        if cv2_img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        results = model.predict(cv2_img, conf=0.5, device="cpu")
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            label = model.names[cls]
            if label in mode_classes.get(mode, []):
                conf = float(box.conf[0].item())
                detections.append({"label": label, "confidence": round(conf, 3), "box": [round(x1), round(y1), round(x2), round(y2)]})

        ocr_results = reader.readtext(cv2_img)
        detected_texts = [text.strip() for (bbox, text, confidence) in ocr_results if confidence > 0.7 and len(text.strip()) > 2 and text.isprintable()]
        
        sentence = ""
        if detections:
            object_list = ", ".join([d["label"] for d in detections])
            sentence += f"Detected objects: {object_list}. "
        if detected_texts:
            combined_text = " ".join(detected_texts)
            sentence += f"The text reads: {combined_text}."

        if sentence:
            speak(sentence)

        return {"detections": detections, "texts": detected_texts}

    except Exception as e:
        logger.error(f"Detection Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/start_gps_navigation")
async def start_gps_navigation(request: GPSRequest):
    try:
        origin = "Vishwakarma Institute of Technology, Kondhwa Campus"
        destination = request.destination
        directions_result = gmaps.directions(origin, destination, mode="walking")
        if not directions_result:
            raise HTTPException(status_code=404, detail="No route found")

        steps = directions_result[0]['legs'][0]['steps']
        directions_text = [re.sub('<[^<]+?>', '', step['html_instructions']) for step in steps]
        return {"instructions": directions_text}
    
    except Exception as e:
        logger.error(f"GPS Error: {e}")
        raise HTTPException(status_code=500, detail=f"Navigation failed: {str(e)}")

@app.post("/start_face_detection")
async def start_face_detection(face_data: FaceData):
    try:
        frame = base64_to_cv2_image(face_data.image_base64)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = DeepFace.analyze(frame_gray, actions=['age'], enforce_detection=False)
        if not faces:
            speak("No face detected")
            return {"result": "No face detected"}

        for face in faces:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            roi_gray = frame_gray[y:y+h, x:x+w]
            result = DeepFace.verify(img1_path=reference_img_gray, img2_path=roi_gray, enforce_detection=False)
            if result["verified"]:
                message = f"{REFERENCE_NAME} detected."
                speak(message)
                return {"result": message}
            else:
                speak("Unknown person detected")
                return {"result": "Unknown person"}

    except Exception as e:
        logger.error(f"Face Detection Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/send_sos")
async def send_sos(request: SOSRequest):
    try:
        message_text = f"Emergency! User is at {request.location}"
        speak(message_text)
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml=f'<Response><Say>{message_text}</Say></Response>',
            to=USER_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER
        )
        return {"status": "SOS call placed successfully."}
    except Exception as e:
        logger.error(f"SOS Error: {e}")
        raise HTTPException(status_code=500, detail=f"SOS Failed: {str(e)}")
