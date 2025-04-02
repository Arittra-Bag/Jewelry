import cv2
import torch
import os
import time
import numpy as np
import sqlite3
import uuid
import dlib
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from mtcnn import MTCNN
import imutils

# Change 1
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

# Load models
face_model = YOLO("yolov8x-face-lindevs.pt")
mtcnn_detector = MTCNN()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Initialize database connection
conn = init_database()
cursor = conn.cursor()

def detect_motion(frame1, frame2, threshold=30):
    """Detect motion between two frames"""
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference between frames
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate motion score
    motion_score = np.sum(thresh) / thresh.size
    return motion_score > 0.01  # Return True if significant motion detected

def detect_eyes(face_img):
    """Enhanced eye detection with multiple checks"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # OpenCV eye detection
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
        
        # Dlib eye detection
        rect = dlib.rectangle(0, 0, face_img.shape[1], face_img.shape[0])
        landmarks = predictor(gray, rect)
        
        # Get eye landmarks
        left_eye = landmarks.parts()[36:42]
        right_eye = landmarks.parts()[42:48]
        
        # Check eye size ratio
        left_eye_size = np.linalg.norm(left_eye[0] - left_eye[3])
        right_eye_size = np.linalg.norm(right_eye[0] - right_eye[3])
        eye_size_ratio = min(left_eye_size, right_eye_size) / max(left_eye_size, right_eye_size)
        
        # Check eye position
        left_eye_center = np.mean([(p.x, p.y) for p in left_eye], axis=0)
        right_eye_center = np.mean([(p.x, p.y) for p in right_eye], axis=0)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        face_width = face_img.shape[1]
        eye_distance_ratio = eye_distance / face_width
        
        # Verify both eyes are detected and properly positioned
        if (len(eyes) >= 2 and 
            len(left_eye) == 6 and 
            len(right_eye) == 6 and
            eye_size_ratio > 0.8 and  # Eyes should be similar in size
            0.2 < eye_distance_ratio < 0.5):  # Eyes should be properly spaced
            return True
        return False
    except Exception as e:
        print(f"Eye detection error: {e}")
        return False

def detect_faces_multi_model(frame):
    """Detect faces using multiple models and combine results"""
    faces = []
    
    # YOLOv8 detection
    yolo_results = face_model(frame)
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            faces.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
    
    # MTCNN detection
    mtcnn_faces = mtcnn_detector.detect_faces(frame)
    for face in mtcnn_faces:
        x, y, w, h = face['box']
        faces.append((x, y, w, h))
    
    # OpenCV cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    for (x, y, w, h) in cv_faces:
        faces.append((x, y, w, h))
    
    # Merge overlapping detections
    return merge_overlapping_faces(faces)

def merge_overlapping_faces(faces, overlap_thresh=0.3):
    """Merge overlapping face detections"""
    if not faces:
        return []
    
    # Convert to numpy array for easier processing
    faces = np.array(faces)
    
    # Calculate areas
    areas = faces[:, 2] * faces[:, 3]
    
    # Sort by area (largest first)
    idxs = np.argsort(areas)[::-1]
    
    # Initialize list of picked indexes
    pick = []
    
    while len(idxs) > 0:
        # Grab the last index and add it to our list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find the intersection
        xx1 = np.maximum(faces[i][0], faces[idxs[:last]][:, 0])
        yy1 = np.maximum(faces[i][1], faces[idxs[:last]][:, 1])
        xx2 = np.minimum(faces[i][0] + faces[i][2], faces[idxs[:last]][:, 0] + faces[idxs[:last]][:, 2])
        yy2 = np.minimum(faces[i][1] + faces[i][3], faces[idxs[:last]][:, 1] + faces[idxs[:last]][:, 3])
        
        # Compute the ratio of overlap
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[idxs[:last]]
        
        # Delete all indexes from the index list that have high overlap
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    return faces[pick].tolist()

def verify_facial_features(face_img):
    """Verify presence of eyes, nose, and lips using facial landmarks"""
    try:
        # Convert to grayscale for dlib
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Detect facial landmarks
        rect = dlib.rectangle(0, 0, face_img.shape[1], face_img.shape[0])
        landmarks = predictor(gray, rect)
        
        # Check for eyes (landmarks 36-47)
        left_eye = landmarks.parts()[36:42]
        right_eye = landmarks.parts()[42:48]
        
        # Check for nose (landmarks 27-35)
        nose = landmarks.parts()[27:35]
        
        # Check for lips (landmarks 48-68)
        lips = landmarks.parts()[48:68]
        
        # Verify all features are present and properly positioned
        if (len(left_eye) == 6 and len(right_eye) == 6 and 
            len(nose) == 8 and len(lips) == 20):
            return True
        return False
    except Exception as e:
        print(f"Facial feature verification error: {e}")
        return False

def verify_face(face_img):
    """Enhanced face verification using multiple strict checks"""
    try:
        # Check image size and quality
        if face_img.shape[0] < 50 or face_img.shape[1] < 50:
            print("Face image too small")
            return False
            
        # Check image blur
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Threshold for blur detection
            print("Image too blurry")
            return False
        
        # First check using OpenCV cascade
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) == 0:
            print("No face detected by OpenCV")
            return False
        
        # Check face aspect ratio
        x, y, w, h = faces[0]
        aspect_ratio = float(w)/h
        if not (0.5 <= aspect_ratio <= 1.5):  # Normal face aspect ratio range
            print("Invalid face aspect ratio")
            return False
        
        # Check face position in frame
        if x < 5 or y < 5 or x + w > face_img.shape[1] - 5 or y + h > face_img.shape[0] - 5:
            print("Face too close to frame edges")
            return False
        
        # Verify eyes with multiple methods
        if not detect_eyes(face_img):
            print("No eyes detected")
            return False
        
        # Verify facial features
        if not verify_facial_features(face_img):
            print("Missing facial features")
            return False
        
        # Check face brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        if brightness < 40 or brightness > 220:  # Too dark or too bright
            print("Invalid face brightness")
            return False
            
        if contrast < 20:  # Too low contrast
            print("Invalid face contrast")
            return False
        
        # Check for face symmetry
        if not check_face_symmetry(face_img):
            print("Face not symmetrical enough")
            return False
        
        return True
    except Exception as e:
        print(f"Face verification error: {e}")
        return False

def check_face_symmetry(face_img):
    """Check face symmetry"""
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Get face landmarks
        rect = dlib.rectangle(0, 0, face_img.shape[1], face_img.shape[0])
        landmarks = predictor(gray, rect)
        
        # Get key facial points
        left_eye = np.mean([(p.x, p.y) for p in landmarks.parts()[36:42]], axis=0)
        right_eye = np.mean([(p.x, p.y) for p in landmarks.parts()[42:48]], axis=0)
        nose = np.mean([(p.x, p.y) for p in landmarks.parts()[27:35]], axis=0)
        mouth_left = np.mean([(p.x, p.y) for p in landmarks.parts()[48:52]], axis=0)
        mouth_right = np.mean([(p.x, p.y) for p in landmarks.parts()[52:56]], axis=0)
        
        # Calculate face center
        face_center = np.mean([left_eye, right_eye, nose], axis=0)
        
        # Check symmetry of eyes
        eye_symmetry = abs(left_eye[0] - (2 * face_center[0] - right_eye[0])) < 10
        
        # Check symmetry of mouth
        mouth_symmetry = abs(mouth_left[0] - (2 * face_center[0] - mouth_right[0])) < 10
        
        # Check vertical alignment
        vertical_alignment = abs(left_eye[1] - right_eye[1]) < 5
        
        return eye_symmetry and mouth_symmetry and vertical_alignment
    except Exception as e:
        print(f"Symmetry check error: {e}")
        return False

def show_verification_window(face_img):
    """Show verification window and perform detailed face scan"""
    # Create verification window
    verification_window = np.zeros((500, 600, 3), dtype=np.uint8)
    cv2.namedWindow("Face Verification")
    
    # Display face image
    display_face = cv2.resize(face_img, (300, 300))
    y_offset = 50
    x_offset = (verification_window.shape[1] - display_face.shape[1]) // 2
    verification_window[y_offset:y_offset+display_face.shape[0], 
                     x_offset:x_offset+display_face.shape[1]] = display_face
    
    # Add title
    cv2.putText(verification_window, "Face Verification in Progress",
              (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Initialize verification results
    verification_results = []
    
    # Perform verification checks
    try:
        # Check image quality
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_check = "Image Quality: " + ("PASS" if laplacian_var >= 100 else "FAIL")
        verification_results.append((quality_check, laplacian_var >= 100))
        
        # Check face detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        face_check = "Face Detection: " + ("PASS" if len(faces) > 0 else "FAIL")
        verification_results.append((face_check, len(faces) > 0))
        
        # Check eyes
        eye_detected = detect_eyes(face_img)
        eye_check = "Eye Detection: " + ("PASS" if eye_detected else "FAIL")
        verification_results.append((eye_check, eye_detected))
        
        # Check facial features
        features_detected = verify_facial_features(face_img)
        feature_check = "Facial Features: " + ("PASS" if features_detected else "FAIL")
        verification_results.append((feature_check, features_detected))
        
        # Check symmetry
        symmetry = check_face_symmetry(face_img)
        symmetry_check = "Face Symmetry: " + ("PASS" if symmetry else "FAIL")
        verification_results.append((symmetry_check, symmetry))
        
        # Display verification results
        y_pos = 380
        for result, passed in verification_results:
            color = (0, 255, 0) if passed else (0, 0, 255)
            cv2.putText(verification_window, result, (50, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25
        
        # Overall verification result
        all_passed = all(passed for _, passed in verification_results)
        result_text = "VERIFICATION SUCCESSFUL" if all_passed else "VERIFICATION FAILED"
        result_color = (0, 255, 0) if all_passed else (0, 0, 255)
        cv2.putText(verification_window, result_text, (150, y_pos + 25),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
        
        # Add instruction
        if all_passed:
            cv2.putText(verification_window, "Press 'r' to register or 'q' to cancel",
                      (150, y_pos + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(verification_window, "Press any key to close",
                      (200, y_pos + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show window and wait for key
        cv2.imshow("Face Verification", verification_window)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Face Verification")
        
        # Return True if verification passed and user pressed 'r'
        return all_passed and key == ord('r')
        
    except Exception as e:
        print(f"Verification window error: {e}")
        cv2.destroyWindow("Face Verification")
        return False

def show_registration_window(face_img):
    """Show registration window for entering customer details"""
    registration_window = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.namedWindow("Register New Customer")
    
    # Display face image
    display_face = cv2.resize(face_img, (200, 200))
    y_offset = (registration_window.shape[0] - display_face.shape[0]) // 4
    x_offset = (registration_window.shape[1] - display_face.shape[1]) // 2
    registration_window[y_offset:y_offset+display_face.shape[0], 
                     x_offset:x_offset+display_face.shape[1]] = display_face
    
    # Add title
    cv2.putText(registration_window, "New Customer Registration",
              (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add text field label
    cv2.putText(registration_window, "Enter Customer Name:",
              (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Add buttons
    cv2.rectangle(registration_window, (100, 320), (200, 350), (100, 100, 100), -1)
    cv2.rectangle(registration_window, (220, 320), (320, 350), (100, 100, 100), -1)
    cv2.putText(registration_window, "Register", (110, 340),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(registration_window, "Cancel", (240, 340),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Register New Customer", registration_window)

def show_scanning_window(face_img):
    """Show scanning animation and verify face before registration"""
    # Create scanning window
    scan_window = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.namedWindow("Scanning Face")
    
    # Display face image
    display_face = cv2.resize(face_img, (200, 200))
    y_offset = (scan_window.shape[0] - display_face.shape[0]) // 4
    x_offset = (scan_window.shape[1] - display_face.shape[1]) // 2
    scan_window[y_offset:y_offset+display_face.shape[0], 
               x_offset:x_offset+display_face.shape[1]] = display_face
    
    # Add scanning line animation
    line_pos = 0
    direction = 1
    scanning = True
    start_time = time.time()
    
    # Quick initial check using MTCNN (lightweight model)
    mtcnn_result = mtcnn_detector.detect_faces(face_img)
    
    while scanning and (time.time() - start_time) < 3:  # Max 3 seconds scan
        scan_frame = scan_window.copy()
        
        # Draw scanning line
        cv2.line(scan_frame, (x_offset, y_offset + line_pos),
                (x_offset + display_face.shape[1], y_offset + line_pos),
                (0, 255, 0), 2)
        
        # Update line position
        line_pos += 2 * direction
        if line_pos >= display_face.shape[0] or line_pos <= 0:
            direction *= -1
        
        # Add scanning text
        cv2.putText(scan_frame, "Scanning for face...",
                  (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Scanning Face", scan_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
            scanning = False
            break
    
    # Show final result
    result_frame = scan_window.copy()
    
    if len(mtcnn_result) > 0:  # Face detected
        cv2.putText(result_frame, "Face Detected!",
                  (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result_frame, "Press ENTER to continue",
                  (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Scanning Face", result_frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Scanning Face")
        return key == 13  # Return True if ENTER pressed
    else:
        cv2.putText(result_frame, "No Face Detected",
                  (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(result_frame, "Press any key to close",
                  (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Scanning Face", result_frame)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyWindow("Scanning Face")
        return False

def mouse_callback(event, x, y, flags, param):
    """Handle mouse click events"""
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, faces, face_cascade = param
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        for (fx, fy, fw, fh) in detected_faces:
            if fx <= x <= fx + fw and fy <= y <= fy + fh:
                face_img = frame[fy:fy+fh, fx:fx+fw]
                
                # First show scanning window
                if show_scanning_window(face_img):
                    # Only proceed with verification if scanning detected a face
                    if verify_face(face_img):
                        # Show verification window
                        if show_verification_window(face_img):
                            # Only show registration window if verification passed
                            show_registration_window(face_img)
                    else:
                        print("Face verification failed - Registration not allowed")
                else:
                    print("No face detected in image - Registration not allowed")

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

def detect_motion_direction(frame1, frame2, threshold=30):
    """Detect motion direction between two frames"""
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference between frames
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate motion score
    motion_score = np.sum(thresh) / thresh.size
    
    # Calculate horizontal motion
    left_half = thresh[:, :thresh.shape[1]//2]
    right_half = thresh[:, thresh.shape[1]//2:]
    left_motion = np.sum(left_half) / left_half.size
    right_motion = np.sum(right_half) / right_half.size
    
    # Determine motion direction
    if motion_score > 0.01:  # Significant motion detected
        if left_motion > right_motion * 1.5:
            return "left"
        elif right_motion > left_motion * 1.5:
            return "right"
        else:
            return "both"
    return None

def extract_and_save_features(face_img, cursor, conn):
    """Enhanced feature extraction and saving with additional checks"""
    try:
        # Verify face with strict checks first
        if not verify_face(face_img):
            print("Face verification failed - Cannot register")
            return None
            
        # Extract features only if face verification passed
        features = extract_face_features(face_img)
        if features is None:
            print("Feature extraction failed")
            return None
            
        # Check if similar face already exists
        cursor.execute("SELECT customer_id, face_features FROM customers")
        found_match = False
        
        for row in cursor.fetchall():
            db_customer_id, db_features = row
            if compare_features(features, db_features):
                found_match = True
                # Update last seen time
                current_time = datetime.now()
                cursor.execute("""
                    UPDATE customers 
                    SET last_seen = ?, visit_count = visit_count + 1
                    WHERE customer_id = ?
                """, (current_time, db_customer_id))
                conn.commit()
                return db_customer_id
        
        if not found_match:
            # Save new face
            customer_id = str(uuid.uuid4())[:8]
            current_time = datetime.now()
            
            # Save face image
            face_path = os.path.join('detected_faces', f'face_{customer_id}.jpg')
            cv2.imwrite(face_path, face_img)
            
            # Save to database
            cursor.execute("""
                INSERT INTO customers (customer_id, face_features, first_seen, last_seen)
                VALUES (?, ?, ?, ?)
            """, (customer_id, features.tobytes(), current_time, current_time))
            conn.commit()
            return customer_id
    except Exception as e:
        print(f"Feature extraction and saving error: {e}")
        return None

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
    
    print("Starting face detection... Press 'q' to quit")
    print("Move side to side to register your face")
    print("Click on a face to verify if it's valid")
    print("Face will only be detected when motion and eyes are detected")
    
    # Initialize motion detection
    ret, prev_frame = cap.read()
    last_feature_extraction_time = 0
    feature_extraction_cooldown = 2  # seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect motion and direction
        motion_direction = detect_motion_direction(prev_frame, frame)
        prev_frame = frame.copy()
        
        current_time = time.time()
        
        if motion_direction in ["left", "right", "both"]:
            # Detect faces using multiple models
            faces = detect_faces_multi_model(frame)
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                
                # Only process if eyes are detected
                if detect_eyes(face_img):
                    # Extract and save features if enough time has passed
                    if current_time - last_feature_extraction_time >= feature_extraction_cooldown:
                        customer_id = extract_and_save_features(face_img, cursor, conn)
                        if customer_id:
                            last_feature_extraction_time = current_time
                            display_text = f"Registered: {customer_id}"
                            color = (0, 255, 0)
                        else:
                            display_text = "Processing..."
                            color = (255, 165, 0)
                    else:
                        display_text = "Cooldown..."
                        color = (255, 165, 0)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, display_text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    # Draw yellow rectangle for faces without detected eyes
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, "No eyes detected", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Display motion direction
        if motion_direction:
            cv2.putText(frame, f"Motion: {motion_direction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection', frame)
        
        # Set up mouse callback
        cv2.setMouseCallback('Face Detection', mouse_callback, (frame, faces, face_cascade))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
