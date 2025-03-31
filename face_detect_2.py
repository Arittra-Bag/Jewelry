import cv2
import torch
import os
import time
import numpy as np
import sqlite3
import uuid
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize database
def init_database():
    """Initialize database and create required directories"""
    os.makedirs('detected_faces', exist_ok=True)
    
    conn = sqlite3.connect('jewelry_shop.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        face_features BLOB,
        name TEXT,
        first_seen DATETIME,
        last_seen DATETIME,
        visit_count INTEGER DEFAULT 1
    )
    ''')
    conn.commit()
    return conn

# Load YOLOv8 model for face detection
face_model = YOLO("yolov8x-face-lindevs.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Initialize database connection
conn = init_database()
cursor = conn.cursor()

def test_camera():
    """Test camera availability and return working camera index"""
    for camera_index in range(2):  # Test first two camera indices
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return camera_index
    return None

def capture_reference_face():
    """Capture the first reference face"""
    print("No faces in database. Let's capture a reference face!")
    print("Position yourself in front of the camera and press 'c' to capture")
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        # Draw rectangle around detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Capture Reference Face', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            features = extract_face_features(face_img)
            
            if features is not None:
                cap.release()
                cv2.destroyAllWindows()
                return face_img, features
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return None, None

def extract_face_features(face_img):
    """Extract basic features from face image"""
    try:
        # Resize to standard size
        face_img = cv2.resize(face_img, (100, 100))
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        # Flatten and normalize
        features = gray.flatten().astype(np.float32)
        features = features / np.linalg.norm(features)
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def compare_features(features1, features2, threshold=0.85):
    """Compare two feature vectors"""
    try:
        if features1 is None or features2 is None:
            return False
            
        if isinstance(features1, bytes):
            features1 = np.frombuffer(features1, dtype=np.float32)
        if isinstance(features2, bytes):
            features2 = np.frombuffer(features2, dtype=np.float32)
        
        similarity = np.dot(features1, features2)
        return similarity > threshold
    except Exception as e:
        print(f"Comparison error: {e}")
        return False

def main():
    # Initialize database
    conn = init_database()
    cursor = conn.cursor()
    
    # Check if database is empty
    cursor.execute("SELECT COUNT(*) FROM customers")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("Database is empty. Capturing reference face...")
        ref_face_img, ref_features = capture_reference_face()
        
        if ref_features is not None:
            # Save reference face
            customer_id = str(uuid.uuid4())[:8]
            current_time = datetime.now()
            
            # Save face image
            face_path = os.path.join('detected_faces', f'face_{customer_id}.jpg')
            cv2.imwrite(face_path, ref_face_img)
            
            # Save to database
            cursor.execute("""
                INSERT INTO customers (customer_id, face_features, first_seen, last_seen)
                VALUES (?, ?, ?, ?)
            """, (customer_id, ref_features.tobytes(), current_time, current_time))
            conn.commit()
            print(f"Reference face saved with ID: {customer_id}")
    
    # Continue with regular face detection
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Starting face detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            features = extract_face_features(face_img)
            
            if features is not None:
                # Compare with stored faces
                cursor.execute("SELECT customer_id, face_features FROM customers")
                found_match = False
                
                for row in cursor.fetchall():
                    db_customer_id, db_features = row
                    if compare_features(features, db_features):
                        found_match = True
                        display_text = f"Match: {db_customer_id}"
                        color = (0, 255, 0)
                        break
                
                if not found_match:
                    display_text = "Unknown Face"
                    color = (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, display_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
