import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTreeWidget, 
                            QTreeWidgetItem, QFrame, QComboBox, QLineEdit,
                            QMessageBox, QDialog, QFileDialog, QScrollArea, QButtonGroup, QRadioButton, QFormLayout)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import cv2
import sqlite3
import os
import uuid
from datetime import datetime
import numpy as np
import time
import io
import requests
import tempfile
from gradio_client import Client
import webbrowser
from PIL import Image, ImageQt

class JewelryShopDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jewelry Shop Security System")
        self.setGeometry(100, 100, 1200, 800)
        
        self.frame_width = 640
        self.frame_height = 480
        
        self.registration_dialog = None
        self.detected_faces = []
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left and right panels
        self.left_panel = QWidget()
        self.right_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        right_layout = QVBoxLayout(self.right_panel)
        
        # Setup camera frame
        self.setup_camera_widget(left_layout)
        
        # Setup customer list
        self.setup_customer_list(right_layout)
        
        # Add panels to main layout
        main_layout.addWidget(self.left_panel)
        main_layout.addWidget(self.right_panel)
        
        # Initialize databases and camera
        self.setup_database()
        self.setup_inventory_database()
        self.setup_camera()
        self.load_existing_customers()
        
        # Timer for camera updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(10)

    def setup_camera_widget(self, layout):
        camera_container = QFrame()
        camera_container.setFrameStyle(QFrame.Shape.Box)
        camera_layout = QVBoxLayout(camera_container)
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(self.frame_width, self.frame_height)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.mousePressEvent = self.on_camera_click
        camera_layout.addWidget(self.camera_label)
        
        # Registration controls
        reg_widget = QWidget()
        reg_layout = QHBoxLayout(reg_widget)
        
        name_label = QLabel("Name:")
        self.name_input = QLineEdit()
        self.register_button = QPushButton("Register Face")
        self.register_button.clicked.connect(self.register_current_face)
        
        reg_layout.addWidget(name_label)
        reg_layout.addWidget(self.name_input)
        reg_layout.addWidget(self.register_button)
        
        camera_layout.addWidget(reg_widget)
        layout.addWidget(camera_container)

    def setup_customer_list(self, layout):
        # Customer list container
        list_container = QFrame()
        list_container.setFrameStyle(QFrame.Shape.Box)
        list_layout = QVBoxLayout(list_container)
        
        # Tree widget for customer list
        self.customer_tree = QTreeWidget()
        self.customer_tree.setHeaderLabels(['ID', 'Name', 'Entry Time', 'Exit Time', 'Status', 'Visits'])
        self.customer_tree.itemDoubleClicked.connect(self.edit_customer)
        
        # Set column widths
        self.customer_tree.setColumnWidth(0, 50)
        self.customer_tree.setColumnWidth(1, 100)
        self.customer_tree.setColumnWidth(2, 150)
        self.customer_tree.setColumnWidth(3, 150)
        self.customer_tree.setColumnWidth(4, 80)
        self.customer_tree.setColumnWidth(5, 50)
        
        list_layout.addWidget(self.customer_tree)
        
        # Control buttons
        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)
        
        buttons = [
            ("Edit Selected", self.edit_customer),
            ("Exit Customer", self.exit_customer),
            ("Past Records", self.show_past_records),
            ("Show Inventory", self.show_inventory),
            ("Delete Selected", self.delete_customer)
        ]
        
        for text, callback in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            button_layout.addWidget(btn)
        
        list_layout.addWidget(button_widget)
        layout.addWidget(list_container)

    def setup_database(self):
        """Setup database with customers and purchases tables"""
        try:
            self.conn = sqlite3.connect('jewelry_shop.db')
            cursor = self.conn.cursor()
            
            # Create customers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    face_encoding BLOB,
                    entry_time DATETIME,
                    exit_time DATETIME,
                    visit_count INTEGER DEFAULT 1
                )
            """)
            
            # Create purchases table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS purchases (
                    purchase_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER,
                    product_id TEXT,
                    product_name TEXT,
                    product_price REAL,
                    product_image BLOB,
                    purchase_time DATETIME,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                )
            """)
            
            self.conn.commit()
            
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to setup database: {e}")

    def setup_inventory_database(self):
        """Setup inventory database"""
        try:
            self.inventory_conn = sqlite3.connect('jewelry_inventory.db')
            cursor = self.inventory_conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inventory (
                    product_id TEXT PRIMARY KEY,
                    product_name TEXT NOT NULL,
                    product_image BLOB,
                    price REAL NOT NULL,
                    quantity INTEGER NOT NULL
                )
            """)
            
            # Check if table is empty and populate with sample data
            cursor.execute("SELECT COUNT(*) FROM inventory")
            if cursor.fetchone()[0] == 0:
                sample_items = [
                    ('J001', 'Gold Necklace', 'jewel1.jpg', 599.99, 5),
                    ('J002', 'Diamond Ring', None, 1299.99, 3),
                    ('J003', 'Silver Bracelet', None, 199.99, 8),
                    ('J004', 'Pearl Earrings', None, 149.99, 6),
                    ('J005', 'Emerald Pendant', None, 799.99, 4),
                    ('J006', 'Ruby Ring', None, 999.99, 2),
                    ('J007', 'Sapphire Bracelet', None, 399.99, 7),
                    ('J008', 'Gold Chain', None, 349.99, 5),
                    ('J009', 'Diamond Studs', None, 699.99, 3),
                    ('J010', 'Silver Anklet', None, 99.99, 10)
                ]
                cursor.executemany("""
                    INSERT INTO inventory (product_id, product_name, product_image, price, quantity)
                    VALUES (?, ?, ?, ?, ?)
                """, sample_items)
                
            self.inventory_conn.commit()
            
        except Exception as e:
            QMessageBox.critical(self, "Inventory Database Error", f"Failed to setup inventory database: {e}")
        
    def setup_camera(self):
        """Setup camera and face detection"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera")
            return
            
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def update_camera(self):
        """Update camera feed with clear status indicators"""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            self.current_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            self.detected_faces = []
            cursor = self.conn.cursor()
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                
                try:
                    current_features = self.extract_face_features(face_img)
                    is_known = False
                    
                    cursor.execute("""
                        SELECT customer_id, name, face_encoding, exit_time
                        FROM customers c1
                        WHERE customer_id IN (
                            SELECT MAX(customer_id) 
                            FROM customers 
                            GROUP BY name
                        )
                    """)
                    customers = cursor.fetchall()
                    
                    for customer_id, name, stored_features, exit_time in customers:
                        if stored_features is not None:
                            stored_features = np.frombuffer(stored_features, dtype=np.float64)
                            if self.compare_features(current_features, stored_features):
                                is_known = True
                                if exit_time is not None:
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 2)
                                    cv2.putText(frame, f"Click to Check-in: {name}", (x, y-10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                                else:
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Checked-in: {name}", (x, y-10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                break
                    
                    if not is_known:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Click to Register New", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    self.detected_faces.append({
                        'bbox': (x, y, w, h),
                        'img': face_img.copy(),
                        'features': current_features
                    })
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
            
            # Convert frame to Qt format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def extract_face_features(self, face_img):
        """Extract features from face image"""
        face_img = cv2.resize(face_img, (128, 128))
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        features = gray.flatten().astype(np.float64)
        features = features / np.linalg.norm(features)
        return features

    def compare_features(self, features1, features2, threshold=0.85):
        """Compare two sets of face features"""
        if features1 is None or features2 is None:
            return False
        if features1.shape != features2.shape:
            return False
        similarity = np.dot(features1, features2)
        return similarity > threshold

    def on_camera_click(self, event):
        """Handle camera click with separate flows for new and returning customers"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            QMessageBox.warning(self, "Warning", "No camera frame available")
            return
        
        # Get click coordinates
        click_x = event.pos().x()
        click_y = event.pos().y()
        
        # Convert coordinates to original frame size
        scale_x = self.frame_width / self.camera_label.width()
        scale_y = self.frame_height / self.camera_label.height()
        
        orig_x = int(click_x * scale_x)
        orig_y = int(click_y * scale_y)
        
        # Check if click is within any detected face
        for face_data in self.detected_faces:
            x, y, w, h = face_data['bbox']
            if x <= orig_x <= x + w and y <= orig_y <= y + h:
                try:
                    current_features = face_data['features']
                    cursor = self.conn.cursor()
                    
                    cursor.execute("""
                        SELECT c1.customer_id, c1.name, c1.face_encoding, c1.exit_time
                        FROM customers c1
                        WHERE c1.customer_id IN (
                            SELECT MAX(customer_id)
                            FROM customers
                            GROUP BY name
                        )
                    """)
                    customers = cursor.fetchall()
                    
                    for customer_id, name, stored_features, exit_time in customers:
                        if stored_features is not None:
                            stored_features = np.frombuffer(stored_features, dtype=np.float64)
                            if self.compare_features(current_features, stored_features):
                                if exit_time is None:
                                    QMessageBox.information(self, "Already Checked In", 
                                                          f"{name} is already checked in!")
                                    return
                                else:
                                    if self.verify_returning_customer(face_data['img']):
                                        self.check_in_customer(name, face_data)
                                    return
                    
                    if self.verify_new_customer(face_data['img']):
                        self.show_registration_dialog(face_data)
                    return
                    
                except Exception as e:
                    print(f"Error checking customer: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to process face: {e}")
                    return
        
        QMessageBox.information(self, "Info", "Please click on a detected face")

    def load_existing_customers(self):
        """Load and display customer list with improved status and visit tracking"""
        try:
            self.customer_tree.clear()
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT 
                    customer_id, 
                    name, 
                    entry_time, 
                    exit_time, 
                    visit_count,
                    (SELECT COUNT(*) FROM customers c2 
                     WHERE c2.name = c1.name) as total_visits
                FROM customers c1
                ORDER BY 
                    CASE WHEN exit_time IS NULL THEN 0 ELSE 1 END,
                    entry_time DESC
            """)
            
            for row in cursor.fetchall():
                customer_id, name, entry_time, exit_time, visit_count, total_visits = row
                
                status = "In Store" if exit_time is None else "Left"
                
                try:
                    entry_time = datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %I:%M %p') if entry_time else "-"
                except:
                    entry_time = "-"
                
                try:
                    exit_time = datetime.strptime(exit_time, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %I:%M %p') if exit_time else "-"
                except:
                    exit_time = "-"
                
                item = QTreeWidgetItem([
                    str(customer_id),
                    name,
                    entry_time,
                    exit_time,
                    status,
                    f"{visit_count}/{total_visits}"
                ])
                
                if status == "In Store":
                    item.setForeground(4, Qt.GlobalColor.green)
                else:
                    item.setForeground(4, Qt.GlobalColor.gray)
                
                self.customer_tree.addTopLevelItem(item)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load customers: {e}")
            print(f"Loading error: {e}")

    def register_current_face(self):
        """Register the current face"""
        if not self.detected_faces:
            QMessageBox.warning(self, "Warning", "No face detected!")
            return
            
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name!")
            return
            
        face_data = self.detected_faces[0]
        self.register_face(face_data, name)

    def register_face(self, face_data, name):
        """Enhanced register face with better error handling"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT name FROM customers WHERE name = ?", (name,))
            existing = cursor.fetchone()
            
            if existing:
                reply = QMessageBox.question(self, "Warning", 
                    f"A customer named {name} already exists. Would you like to update their record?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                
                if reply == QMessageBox.StandardButton.Yes:
                    cursor.execute("""
                        UPDATE customers 
                        SET face_encoding = ?,
                            entry_time = datetime('now'),
                            exit_time = NULL,
                            visit_count = visit_count + 1
                        WHERE name = ?
                    """, (face_data['features'].tobytes(), name))
                else:
                    return
            else:
                cursor.execute("""
                    INSERT INTO customers 
                    (name, face_encoding, entry_time, visit_count)
                    VALUES (?, ?, datetime('now'), 1)
                """, (name, face_data['features'].tobytes()))
            
            self.conn.commit()
            QMessageBox.information(self, "Success", f"Successfully registered {name}")
            self.name_input.clear()
            self.load_existing_customers()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to register face: {e}")
            print(f"Registration error: {e}")

    # Remove this line:
    # from mtcnn import MTCNN

    # Keep all other imports, and modify the verification functions:

    def verify_new_customer(self, face_img):
        """Verify a new customer using OpenCV face detection"""
        verify_dialog = QDialog(self)
        verify_dialog.setWindowTitle("New Customer Verification")
        verify_dialog.setFixedSize(400, 300)
        
        layout = QVBoxLayout(verify_dialog)
        
        # Image display
        image_label = QLabel()
        face_display = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_display = cv2.resize(face_display, (200, 200))
        qt_image = QImage(face_display.data, face_display.shape[1], face_display.shape[0], 
                        face_display.shape[1] * 3, QImage.Format.Format_RGB888)
        image_label.setPixmap(QPixmap.fromImage(qt_image))
        layout.addWidget(image_label)
        
        # Status label
        status_label = QLabel("Verifying...")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_label)
        
        verify_dialog.show()
        
        # Use OpenCV face detection
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        verified = False
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_width = face_img.shape[1]
            face_ratio = w / face_width
            
            if 0.2 <= face_ratio <= 0.8:  # Reasonable face size ratio
                status_label.setText("Verification Successful!")
                status_label.setStyleSheet("color: green")
                verified = True
            else:
                status_label.setText("Verification Failed")
                status_label.setStyleSheet("color: red")
        else:
            status_label.setText("Verification Failed")
            status_label.setStyleSheet("color: red")
        
        QTimer.singleShot(1000, verify_dialog.close)
        return verified

    def verify_returning_customer(self, face_img):
        """Quick verification for returning customers using OpenCV"""
        verify_dialog = QDialog(self)
        verify_dialog.setWindowTitle("Welcome Back!")
        verify_dialog.setFixedSize(400, 300)
        
        layout = QVBoxLayout(verify_dialog)
        
        # Image display
        image_label = QLabel()
        face_display = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_display = cv2.resize(face_display, (200, 200))
        qt_image = QImage(face_display.data, face_display.shape[1], face_display.shape[0], 
                        face_display.shape[1] * 3, QImage.Format.Format_RGB888)
        image_label.setPixmap(QPixmap.fromImage(qt_image))
        layout.addWidget(image_label)
        
        # Status label
        status_label = QLabel("Verifying...")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_label)
        
        verify_dialog.show()
        
        # Use OpenCV face detection
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        verified = False
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_width = face_img.shape[1]
            face_ratio = w / face_width
            
            if 0.2 <= face_ratio <= 0.8:  # Reasonable face size ratio
                status_label.setText("Welcome Back!")
                status_label.setStyleSheet("color: green")
                verified = True
            else:
                status_label.setText("Verification Failed")
                status_label.setStyleSheet("color: red")
        else:
            status_label.setText("Verification Failed")
            status_label.setStyleSheet("color: red")
        
        QTimer.singleShot(1000, verify_dialog.close)
        return verified

    def check_in_customer(self, name, face_data):
        """Check in a returning customer"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE customers 
                SET face_encoding = ?,
                    entry_time = datetime('now'),
                    exit_time = NULL,
                    visit_count = visit_count + 1
                WHERE name = ?
            """, (face_data['features'].tobytes(), name))
            
            self.conn.commit()
            QMessageBox.information(self, "Success", f"Welcome back, {name}!")
            self.load_existing_customers()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to check in customer: {e}")
            print(f"Check-in error: {e}")

    def delete_customer(self):
        """Delete selected customer"""
        selected_items = self.customer_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a customer to delete")
            return
        
        item = selected_items[0]
        customer_id = int(item.text(0))
        customer_name = item.text(1)
        
        reply = QMessageBox.question(self, "Confirm Delete", 
                                   f"Are you sure you want to delete {customer_name}?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM purchases WHERE customer_id = ?", (customer_id,))
                cursor.execute("DELETE FROM customers WHERE customer_id = ?", (customer_id,))
                self.conn.commit()
                
                self.load_existing_customers()
                QMessageBox.information(self, "Success", f"Successfully deleted {customer_name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete customer: {e}")
                print(f"Delete error: {e}")

    def edit_customer(self):
        """Edit customer with name propagation to all entries"""
        selected_items = self.customer_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a customer to edit")
            return
        
        customer_id = int(selected_items[0].text(0))
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT c1.customer_id, c1.name, c1.entry_time, c1.exit_time, c1.visit_count,
                       (SELECT GROUP_CONCAT(c2.customer_id) 
                        FROM customers c2 
                        WHERE c2.name = c1.name) as related_ids
                FROM customers c1
                WHERE c1.customer_id = ?
            """, (customer_id,))
            customer = cursor.fetchone()
            
            if not customer:
                QMessageBox.critical(self, "Error", "Customer not found")
                return
            
            edit_dialog = QDialog(self)
            edit_dialog.setWindowTitle("Edit Customer")
            edit_dialog.setFixedSize(400, 500)
            
            layout = QVBoxLayout(edit_dialog)
            
            # Warning label
            warning_label = QLabel("Note: Changing the name will update all visits for this customer")
            warning_label.setStyleSheet("color: red")
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)
            
            # Form fields
            form_widget = QWidget()
            form_layout = QVBoxLayout(form_widget)
            
            # Customer ID
            id_layout = QHBoxLayout()
            id_layout.addWidget(QLabel("Customer ID:"))
            id_input = QLineEdit(str(customer[0]))
            id_input.setReadOnly(True)
            id_layout.addWidget(id_input)
            form_layout.addLayout(id_layout)
            
            # Name
            name_layout = QHBoxLayout()
            name_layout.addWidget(QLabel("Name:"))
            name_input = QLineEdit(customer[1])
            name_layout.addWidget(name_input)
            form_layout.addLayout(name_layout)
            
            # Entry Time
            entry_layout = QHBoxLayout()
            entry_layout.addWidget(QLabel("Entry Time:"))
            entry_input = QLineEdit(customer[2])
            entry_layout.addWidget(entry_input)
            form_layout.addLayout(entry_layout)
            
            # Exit Time
            exit_layout = QHBoxLayout()
            exit_layout.addWidget(QLabel("Exit Time:"))
            exit_input = QLineEdit(customer[3] if customer[3] else "")
            exit_layout.addWidget(exit_input)
            form_layout.addLayout(exit_layout)
            
            # Visit Count
            visit_layout = QHBoxLayout()
            visit_layout.addWidget(QLabel("Visit Count:"))
            visit_input = QLineEdit(str(customer[4]))
            visit_layout.addWidget(visit_input)
            form_layout.addLayout(visit_layout)
            
            layout.addWidget(form_widget)
            
            # Related IDs info
            related_ids = customer[5].split(',') if customer[5] else []
            related_label = QLabel(f"Total visits to update: {len(related_ids)}")
            related_label.setStyleSheet("font-style: italic")
            layout.addWidget(related_label)
            
            # Buttons
            button_layout = QHBoxLayout()
            save_button = QPushButton("Save Changes")
            cancel_button = QPushButton("Cancel")
            button_layout.addWidget(save_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)
            
            def save_changes():
                try:
                    new_name = name_input.text().strip()
                    if not new_name:
                        QMessageBox.warning(edit_dialog, "Warning", "Name cannot be empty")
                        return
                    
                    old_name = customer[1]
                    
                    if new_name != old_name:
                        reply = QMessageBox.question(edit_dialog, "Confirm Name Change",
                            f"Are you sure you want to change the name from '{old_name}' to '{new_name}'?\n\n"
                            f"This will update {len(related_ids)} visit records for this customer.",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        
                        if reply == QMessageBox.StandardButton.Yes:
                            cursor.execute("""
                                UPDATE customers 
                                SET name = ?
                                WHERE name = ?
                            """, (new_name, old_name))
                    
                    cursor.execute("""
                        UPDATE customers 
                        SET entry_time = ?,
                            exit_time = ?,
                            visit_count = ?
                        WHERE customer_id = ?
                    """, (
                        entry_input.text(),
                        exit_input.text() if exit_input.text() else None,
                        int(visit_input.text()),
                        customer_id
                    ))
                    
                    self.conn.commit()
                    QMessageBox.information(edit_dialog, "Success", 
                        "Customer information updated successfully\n"
                        f"Updated {len(related_ids)} visit records")
                    edit_dialog.accept()
                    self.load_existing_customers()
                    
                except Exception as e:
                    QMessageBox.critical(edit_dialog, "Error", f"Failed to update customer: {e}")
                    print(f"Update error: {e}")
            
            save_button.clicked.connect(save_changes)
            cancel_button.clicked.connect(edit_dialog.reject)
            
            edit_dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open edit dialog: {e}")
            print(f"Edit dialog error: {e}")

    def show_inventory(self):
        """Show current inventory with Product ID and Quantity"""
        try:
            inventory_dialog = QDialog(self)
            inventory_dialog.setWindowTitle("Current Inventory")
            inventory_dialog.setFixedSize(400, 400)
            
            layout = QVBoxLayout(inventory_dialog)
            
            # Inventory tree widget
            tree = QTreeWidget()
            tree.setHeaderLabels(['Product ID', 'Quantity'])
            tree.setColumnWidth(0, 150)
            tree.setColumnWidth(1, 100)
            layout.addWidget(tree)
            
            # Fetch and display inventory data
            cursor = self.inventory_conn.cursor()
            cursor.execute("SELECT product_id, quantity FROM inventory ORDER BY product_id")
            inventory_items = cursor.fetchall()
            
            for product_id, quantity in inventory_items:
                QTreeWidgetItem(tree, [product_id, str(quantity)])
            
            close_button = QPushButton("Close")
            close_button.clicked.connect(inventory_dialog.accept)
            layout.addWidget(close_button)
            
            inventory_dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to show inventory: {e}")
            print(f"Inventory display error: {e}")

    def show_past_records(self):
        """Show past records for selected customer"""
        selected_items = self.customer_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a customer to view past records")
            return
        
        customer_name = selected_items[0].text(1)
        
        try:
            records_dialog = QDialog(self)
            records_dialog.setWindowTitle(f"Past Records for {customer_name}")
            records_dialog.setFixedSize(1000, 500)
            
            layout = QVBoxLayout(records_dialog)
            
            # Records tree widget
            tree = QTreeWidget()
            tree.setHeaderLabels(['Visit #', 'Entry Time', 'Exit Time', 'Duration', 
                                'Product ID', 'Product', 'Price', 'Product Image', 'Generate'])
            
            # Set column widths
            column_widths = [60, 150, 150, 100, 80, 120, 80, 100, 100]
            for i, width in enumerate(column_widths):
                tree.setColumnWidth(i, width)
            
            layout.addWidget(tree)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT 
                    c.customer_id, 
                    c.entry_time, 
                    c.exit_time, 
                    c.visit_count,
                    p.product_id,
                    p.product_name,
                    p.product_price,
                    p.product_image
                FROM customers c
                LEFT JOIN purchases p ON c.customer_id = p.customer_id
                WHERE c.name = ?
                ORDER BY c.entry_time DESC
            """, (customer_name,))
            
            records = cursor.fetchall()
            
            for record in records:
                customer_id, entry_time, exit_time, visit_count, product_id, product_name, product_price, product_image = record
                
                try:
                    entry_dt = datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S')
                    entry_str = entry_dt.strftime('%Y-%m-%d %I:%M %p')
                    
                    if exit_time:
                        exit_dt = datetime.strptime(exit_time, '%Y-%m-%d %H:%M:%S')
                        exit_str = exit_dt.strftime('%Y-%m-%d %I:%M %p')
                        duration = exit_dt - entry_dt
                        duration_str = str(duration)
                        if duration.total_seconds() < 3600:
                            minutes = int(duration.total_seconds() / 60)
                            duration_str = f"{minutes} minutes"
                    else:
                        exit_str = "-"
                        duration_str = "In Progress"
                    
                    if not any([product_id, product_name, product_price]):
                        product_values = ["-"] * 5
                    else:
                        product_values = [
                            product_id or "-",
                            product_name or "-",
                            f"${product_price:.2f}" if product_price else "-",
                            "View" if product_image else "-",
                            "Generate" if product_image else "-"
                        ]
                    
                    item = QTreeWidgetItem([
                        str(visit_count),
                        entry_str,
                        exit_str,
                        duration_str,
                        *product_values
                    ])
                    
                    if product_image:
                        item.setForeground(7, Qt.GlobalColor.blue)
                        item.setForeground(8, Qt.GlobalColor.blue)
                    
                    tree.addTopLevelItem(item)
                    
                except Exception as e:
                    print(f"Error processing record {customer_id}: {e}")
                    continue
            
            def on_tree_click(item, column):
                if column == 7 and item.text(7) == "View":
                    self.view_image(records[tree.indexOfTopLevelItem(item)][7])
                elif column == 8 and item.text(8) == "Generate":
                    self.generate_image_from_gradio(item.text(4))  # product_id
            
            tree.itemClicked.connect(on_tree_click)
            
            close_button = QPushButton("Close")
            close_button.clicked.connect(records_dialog.accept)
            layout.addWidget(close_button)
            
            records_dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to show past records: {e}")
            print(f"Past records error: {e}")

    def view_image(self, image_data):
        """Display the product image in a new window"""
        try:
            if not image_data:
                QMessageBox.information(self, "Info", "No image data available")
                return
            
            image_dialog = QDialog(self)
            image_dialog.setWindowTitle("Product Image")
            image_dialog.setFixedSize(350, 350)
            
            layout = QVBoxLayout(image_dialog)
            
            image_label = QLabel()
            img = Image.open(io.BytesIO(image_data))
            img = img.resize((300, 300))
            qimg = ImageQt.ImageQt(img)
            pixmap = QPixmap.fromImage(qimg)
            image_label.setPixmap(pixmap)
            layout.addWidget(image_label)
            
            close_button = QPushButton("Close")
            close_button.clicked.connect(image_dialog.accept)
            layout.addWidget(close_button)
            
            image_dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display image: {e}")
            print(f"Image display error: {e}")

    def generate_image_from_gradio(self, product_id):
        """Fetch product image based on product ID and redirect to Gradio"""
        try:
            image_number = int(product_id[1:])
            image_filename = f"jewel{image_number}.jpg"
            image_path = os.path.join("C:\\Users\\aritt\\OneDrive\\Desktop\\Jewelry", image_filename)
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            gradio_url = "http://127.0.0.1:7860/"
            webbrowser.open(gradio_url)
            
            QMessageBox.information(self, "Redirecting",
                f"The browser has been opened to {gradio_url}.\n"
                f"Please manually upload the image located at:\n{image_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process redirection: {e}")
            print(f"Redirection error: {e}")

    def exit_customer(self):
        """Mark selected customer as exited and record purchase details with inventory integration"""
        selected_items = self.customer_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a customer to mark as exited")
            return
        
        item = selected_items[0]
        customer_id = int(item.text(0))
        customer_name = item.text(1)
        status = item.text(4)
        
        if status == "Left":
            QMessageBox.information(self, "Info", f"{customer_name} has already been marked as exited")
            return
        
        exit_dialog = QDialog(self)
        exit_dialog.setWindowTitle(f"Exit Details for {customer_name}")
        exit_dialog.setFixedSize(400, 500)
        
        layout = QVBoxLayout(exit_dialog)
        
        # Purchase information frame
        form_frame = QFrame()
        form_frame.setFrameStyle(QFrame.Shape.Box)
        form_layout = QVBoxLayout(form_frame)
        
        # Purchase radio buttons
        form_layout.addWidget(QLabel("Did the customer make a purchase?"))
        purchase_var = QButtonGroup(exit_dialog)
        yes_radio = QRadioButton("Yes")
        no_radio = QRadioButton("No")
        no_radio.setChecked(True)
        purchase_var.addButton(yes_radio)
        purchase_var.addButton(no_radio)
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(yes_radio)
        radio_layout.addWidget(no_radio)
        form_layout.addLayout(radio_layout)
        
        # Purchase details widget
        purchase_widget = QWidget()
        purchase_layout = QFormLayout(purchase_widget)
        
        # Fetch available product IDs
        try:
            cursor = self.inventory_conn.cursor()
            cursor.execute("SELECT product_id FROM inventory WHERE quantity > 0")
            available_products = [row[0] for row in cursor.fetchall()]
            if not available_products:
                available_products = ["No products available"]
        except Exception as e:
            print(f"Error fetching product IDs: {e}")
            available_products = ["Error loading products"]
        
        # Product selection combo box
        product_combo = QComboBox()
        product_combo.addItems(available_products)
        purchase_layout.addRow("Product ID:", product_combo)
        
        # Product details labels
        product_name_label = QLabel()
        product_price_label = QLabel()
        product_image_label = QLabel()
        purchase_layout.addRow("Product Name:", product_name_label)
        purchase_layout.addRow("Price:", product_price_label)
        purchase_layout.addRow("Image:", product_image_label)
        
        def update_product_details():
            product_id = product_combo.currentText()
            if product_id and product_id not in ["No products available", "Error loading products"]:
                try:
                    cursor = self.inventory_conn.cursor()
                    cursor.execute("""
                        SELECT product_name, price, quantity, product_image 
                        FROM inventory 
                        WHERE product_id = ?
                    """, (product_id,))
                    result = cursor.fetchone()
                    if result:
                        product_name_label.setText(result[0])
                        product_price_label.setText(f"${result[1]:.2f}")
                        if result[2] <= 0:
                            QMessageBox.warning(exit_dialog, "Warning", "This product is out of stock!")
                        
                        if result[3]:
                            img = Image.open(io.BytesIO(result[3]))
                            img = img.resize((150, 150))
                            qimg = ImageQt.ImageQt(img)
                            pixmap = QPixmap.fromImage(qimg)
                            product_image_label.setPixmap(pixmap)
                        else:
                            product_image_label.setText("No image available")
                    else:
                        product_name_label.setText("Not found")
                        product_price_label.setText("N/A")
                        product_image_label.setText("Product not found")
                except Exception as e:
                    QMessageBox.critical(exit_dialog, "Error", f"Failed to fetch product details: {e}")
                    print(f"Error fetching product details for {product_id}: {e}")
            else:
                product_name_label.setText("")
                product_price_label.setText("")
                product_image_label.setText("")
        
        product_combo.currentTextChanged.connect(update_product_details)
        update_product_details()
        
        purchase_widget.setVisible(False)
        form_layout.addWidget(purchase_widget)
        
        def toggle_purchase_details():
            purchase_widget.setVisible(yes_radio.isChecked())
        
        yes_radio.toggled.connect(toggle_purchase_details)
        
        layout.addWidget(form_frame)
        
        # Buttons
        button_layout = QHBoxLayout()
        submit_button = QPushButton("Submit")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(submit_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        def submit_exit():
            try:
                cursor = self.conn.cursor()
                inventory_cursor = self.inventory_conn.cursor()
                
                cursor.execute("""
                    UPDATE customers 
                    SET exit_time = datetime('now')
                    WHERE customer_id = ? AND exit_time IS NULL
                """, (customer_id,))
                
                if yes_radio.isChecked():
                    product_id = product_combo.currentText()
                    if not product_id or product_id in ["No products available", "Error loading products"]:
                        QMessageBox.warning(exit_dialog, "Warning", "Please select a valid Product ID")
                        return
                    
                    inventory_cursor.execute("""
                        SELECT product_name, price, quantity, product_image
                        FROM inventory 
                        WHERE product_id = ?
                    """, (product_id,))
                    product_details = inventory_cursor.fetchone()
                    
                    if not product_details:
                        QMessageBox.warning(exit_dialog, "Warning", "Invalid Product ID")
                        return
                    
                    if product_details[2] <= 0:
                        QMessageBox.warning(exit_dialog, "Warning", "Product is out of stock")
                        return
                    
                    product_name, product_price, quantity, product_image = product_details
                    
                    cursor.execute("""
                        INSERT INTO purchases 
                        (customer_id, product_id, product_name, product_price, product_image, purchase_time)
                        VALUES (?, ?, ?, ?, ?, datetime('now'))
                    """, (customer_id, product_id, product_name, product_price, product_image))
                    
                    inventory_cursor.execute("""
                        UPDATE inventory 
                        SET quantity = quantity - 1
                        WHERE product_id = ?
                    """, (product_id,))
                
                self.conn.commit()
                self.inventory_conn.commit()
                
                if cursor.rowcount > 0:
                    QMessageBox.information(exit_dialog, "Success", 
                                        f"Successfully marked {customer_name} as exited")
                    exit_dialog.accept()
                    self.load_existing_customers()
                else:
                    QMessageBox.warning(exit_dialog, "Warning", 
                                    "Customer record not found or already exited")
                    exit_dialog.reject()
            
            except Exception as e:
                QMessageBox.critical(exit_dialog, "Error", f"Failed to process exit: {e}")
                print(f"Exit error: {e}")
        
        submit_button.clicked.connect(submit_exit)
        cancel_button.clicked.connect(exit_dialog.reject)
        
        exit_dialog.exec()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    window = JewelryShopDashboard()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()