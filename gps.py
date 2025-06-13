import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


image_paths = {

    "Anshul":r"C:\Users\VICTUS\Downloads\IMG_20240130_203504173_DOC.jpg"
   
}


known_faces = {}


for name, path in image_paths.items():
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image {path} not found or invalid format.")
        continue

 
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0] 
        cropped_face = img[y:y+h, x:x+w]
        resized_face = cv2.resize(cropped_face, (100, 100))  
        known_faces[name] = resized_face
    else:
        print(f"No face detected in image: {path}")

print("Known faces loaded successfully:", list(known_faces.keys()))


video_capture = cv2.VideoCapture(0)
print("Press 'Esc' to exit.")

while True:
  
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

  
    detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        cropped_face = gray_frame[y:y+h, x:x+w]
        resized_face = cv2.resize(cropped_face, (100, 100))  

        name = "Unknown" 
        for person_name, known_face in known_faces.items():
            diff = cv2.absdiff(resized_face, known_face)
            score = np.mean(diff)

           
            threshold = 35
            if score < threshold:
                name = person_name
                break

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

   
    cv2.imshow("Face Detection and Matching", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


video_capture.release()
cv2.destroyAllWindows()