import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import sqlite3
import os
import uuid
from datetime import datetime
import numpy as np
import time

class JewelryShopDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Jewelry Shop Security System")
        self.root.geometry("1200x800")
        
        self.frame_width = 640
        self.frame_height = 480
        
        self.registration_dialog = None
        self.detected_faces = []
        
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.left_panel = ttk.Frame(self.main_container)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.setup_database()
        self.create_camera_frame()
        self.setup_camera()
        self.create_customer_list()
        self.load_existing_customers()

    def setup_database(self):
        """Setup database with customers and purchases tables"""
        try:
            self.conn = sqlite3.connect('jewelry_shop.db')
            cursor = self.conn.cursor()
            
            # Customers table
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='customers'
            """)
            if not cursor.fetchone():
                cursor.execute("""
                    CREATE TABLE customers (
                        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        face_encoding BLOB,
                        entry_time DATETIME,
                        exit_time DATETIME,
                        visit_count INTEGER DEFAULT 1
                    )
                """)
            
            # Purchases table
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='purchases'
            """)
            if not cursor.fetchone():
                cursor.execute("""
                    CREATE TABLE purchases (
                        purchase_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        customer_id INTEGER,
                        product_id TEXT,
                        product_name TEXT,
                        product_price REAL,
                        purchase_time DATETIME,
                        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                    )
                """)
            
            self.conn.commit()
                
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to setup database: {e}")

    def setup_camera(self):
        """Setup camera and face detection"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
            
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.update_camera()

    def load_existing_customers(self):
        """Load and display customer list with improved status and visit tracking"""
        try:
            self.tree.delete(*self.tree.get_children())
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
                
                tags = ('exited',) if status == "Left" else ('active',)
                self.tree.insert('', 'end', values=(
                    customer_id,
                    name,
                    entry_time,
                    exit_time,
                    status,
                    f"{visit_count}/{total_visits}"
                ), tags=tags)
            
            self.tree.tag_configure('active', foreground='green')
            self.tree.tag_configure('exited', foreground='gray')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load customers: {e}")
            print(f"Loading error: {e}")

    def create_camera_frame(self):
        """Create camera frame with click functionality"""
        camera_container = ttk.LabelFrame(self.left_panel, text="Live Camera Feed")
        camera_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.camera_canvas = tk.Canvas(
            camera_container,
            width=self.frame_width,
            height=self.frame_height,
            bg='black'
        )
        self.camera_canvas.pack(padx=5, pady=5)
        
        self.camera_canvas.bind('<Button-1>', self.on_camera_click)
        
        reg_frame = ttk.Frame(camera_container)
        reg_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(reg_frame, text="Name:").pack(side=tk.LEFT, padx=5)
        self.manual_name_var = tk.StringVar()
        name_entry = ttk.Entry(reg_frame, textvariable=self.manual_name_var)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.register_button = ttk.Button(
            reg_frame,
            text="Register Face",
            command=self.register_current_face
        )
        self.register_button.pack(side=tk.LEFT, padx=5)

    def create_customer_list(self):
        """Create customer list with edit, delete, exit, and past records buttons"""
        list_container = ttk.LabelFrame(self.right_panel, text="Customer Records")
        list_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('ID', 'Name', 'Entry Time', 'Exit Time', 'Status', 'Visits')
        self.tree = ttk.Treeview(list_container, columns=columns, show='headings')
        
        self.tree.heading('ID', text='ID')
        self.tree.heading('Name', text='Name')
        self.tree.heading('Entry Time', text='Entry Time')
        self.tree.heading('Exit Time', text='Exit Time')
        self.tree.heading('Status', text='Status')
        self.tree.heading('Visits', text='Visits')
        
        self.tree.column('ID', width=50)
        self.tree.column('Name', width=100)
        self.tree.column('Entry Time', width=150)
        self.tree.column('Exit Time', width=150)
        self.tree.column('Status', width=80)
        self.tree.column('Visits', width=50)
        
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        buttons_frame = ttk.Frame(list_container)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        style = ttk.Style()
        style.configure('Edit.TButton', background='#4CAF50')
        style.configure('Delete.TButton', background='#f44336')
        style.configure('Exit.TButton', background='#FF9800')
        style.configure('Records.TButton', background='#2196F3')
        
        ttk.Button(
            buttons_frame, 
            text="Edit Selected", 
            style='Edit.TButton',
            command=self.edit_customer
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame, 
            text="Exit Customer", 
            style='Exit.TButton',
            command=self.exit_customer
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame, 
            text="Past Records", 
            style='Records.TButton',
            command=self.show_past_records
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame, 
            text="Delete Selected", 
            style='Delete.TButton',
            command=self.delete_customer
        ).pack(side=tk.LEFT, padx=5)
        
        self.tree.bind('<Double-1>', lambda e: self.edit_customer())

    def extract_face_features(self, face_img):
        """Extract features from face image"""
        face_img = cv2.resize(face_img, (128, 128))
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        features = gray.flatten().astype(np.float64)
        features = features / np.linalg.norm(features)
        return features

    def update_camera(self):
        """Update camera feed with clear status indicators"""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
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
            
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(cv2_im))
            self.camera_canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
            self.camera_canvas.image = img_tk
            
        self.root.after(10, self.update_camera)

    def compare_features(self, features1, features2, threshold=0.85):
        """Compare two sets of face features"""
        if features1 is None or features2 is None:
            return False
        if features1.shape != features2.shape:
            return False
        similarity = np.dot(features1, features2)
        return similarity > threshold

    def register_current_face(self):
        """Register the current face"""
        if not self.detected_faces:
            messagebox.showwarning("Warning", "No face detected!")
            return
            
        if not self.manual_name_var.get().strip():
            messagebox.showwarning("Warning", "Please enter a name!")
            return
            
        face_data = self.detected_faces[0]
        self.register_face(face_data, self.manual_name_var.get().strip())

    def register_face(self, face_data, name):
        """Enhanced register face with better error handling"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT name FROM customers WHERE name = ?", (name,))
            existing = cursor.fetchone()
            
            if existing:
                if messagebox.askyesno("Warning", 
                    f"A customer named {name} already exists. Would you like to update their record?"):
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
            messagebox.showinfo("Success", f"Successfully registered {name}")
            self.manual_name_var.set("")
            self.load_existing_customers()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to register face: {e}")
            print(f"Registration error: {e}")

    def delete_customer(self):
        """Delete selected customer"""
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a customer to delete")
            return
        
        customer_id = self.tree.item(selected_item[0])['values'][0]
        customer_name = self.tree.item(selected_item[0])['values'][1]
        
        if not messagebox.askyesno("Confirm Delete", 
                                 f"Are you sure you want to delete {customer_name}?"):
            return
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM purchases WHERE customer_id = ?", (customer_id,))
            cursor.execute("DELETE FROM customers WHERE customer_id = ?", (customer_id,))
            self.conn.commit()
            
            self.tree.delete(selected_item)
            messagebox.showinfo("Success", f"Successfully deleted {customer_name}")
            self.load_existing_customers()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete customer: {e}")
            print(f"Delete error: {e}")

    def edit_customer(self):
        """Edit customer with name propagation to all entries"""
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a customer to edit")
            return
        
        customer_id = self.tree.item(selected_item[0])['values'][0]
        
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
                messagebox.showerror("Error", "Customer not found")
                return
            
            dialog = tk.Toplevel(self.root)
            dialog.title("Edit Customer")
            dialog.geometry("400x500")
            dialog.transient(self.root)
            dialog.grab_set()
            
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            x = (dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (dialog.winfo_screenheight() // 2) - (height // 2)
            dialog.geometry(f'+{x}+{y}')
            
            form_frame = ttk.LabelFrame(dialog, text="Edit Customer Details", padding=10)
            form_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            warning_label = ttk.Label(
                form_frame, 
                text="Note: Changing the name will update all visits for this customer",
                foreground='red',
                wraplength=300
            )
            warning_label.grid(row=0, column=0, columnspan=2, pady=10)
            
            ttk.Label(form_frame, text="Customer ID:").grid(row=1, column=0, sticky='e', pady=5)
            id_var = tk.StringVar(value=str(customer[0]))
            ttk.Entry(form_frame, textvariable=id_var, state='readonly').grid(row=1, column=1, sticky='ew', pady=5)
            
            ttk.Label(form_frame, text="Name:").grid(row=2, column=0, sticky='e', pady=5)
            name_var = tk.StringVar(value=customer[1])
            name_entry = ttk.Entry(form_frame, textvariable=name_var)
            name_entry.grid(row=2, column=1, sticky='ew', pady=5)
            
            ttk.Label(form_frame, text="Entry Time:").grid(row=3, column=0, sticky='e', pady=5)
            entry_time_var = tk.StringVar(value=customer[2])
            entry_time_entry = ttk.Entry(form_frame, textvariable=entry_time_var)
            entry_time_entry.grid(row=3, column=1, sticky='ew', pady=5)
            
            ttk.Label(form_frame, text="Exit Time:").grid(row=4, column=0, sticky='e', pady=5)
            exit_time_var = tk.StringVar(value=customer[3] if customer[3] else "")
            exit_time_entry = ttk.Entry(form_frame, textvariable=exit_time_var)
            exit_time_entry.grid(row=4, column=1, sticky='ew', pady=5)
            
            ttk.Label(form_frame, text="Visit Count:").grid(row=5, column=0, sticky='e', pady=5)
            visit_count_var = tk.StringVar(value=str(customer[4]))
            visit_count_entry = ttk.Entry(form_frame, textvariable=visit_count_var)
            visit_count_entry.grid(row=5, column=1, sticky='ew', pady=5)
            
            related_ids = customer[5].split(',') if customer[5] else []
            ttk.Label(form_frame, 
                     text=f"Total visits to update: {len(related_ids)}",
                     font=('Helvetica', 9, 'italic')).grid(row=6, column=0, columnspan=2, pady=10)
            
            def save_changes():
                try:
                    new_name = name_var.get().strip()
                    if not new_name:
                        messagebox.showwarning("Warning", "Name cannot be empty")
                        return
                    
                    old_name = customer[1]
                    
                    if new_name != old_name:
                        if not messagebox.askyesno("Confirm Name Change",
                            f"Are you sure you want to change the name from '{old_name}' to '{new_name}'?\n\n"
                            f"This will update {len(related_ids)} visit records for this customer."):
                            return
                        
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
                        entry_time_var.get(),
                        exit_time_var.get() if exit_time_var.get() else None,
                        int(visit_count_var.get()),
                        customer_id
                    ))
                    
                    self.conn.commit()
                    messagebox.showinfo("Success", 
                        "Customer information updated successfully\n"
                        f"Updated {len(related_ids)} visit records")
                    dialog.destroy()
                    self.load_existing_customers()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to update customer: {e}")
                    print(f"Update error: {e}")
            
            button_frame = ttk.Frame(dialog)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Button(
                button_frame,
                text="Save Changes",
                style='Edit.TButton',
                command=save_changes
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame,
                text="Cancel",
                command=dialog.destroy
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open edit dialog: {e}")
            print(f"Edit dialog error: {e}")

    def on_camera_click(self, event):
        """Handle clicks on the camera feed with automatic recognition"""
        if not self.detected_faces:
            messagebox.showinfo("Info", "No faces detected to register!")
            return
            
        for face_data in self.detected_faces:
            x, y, w, h = face_data['bbox']
            if (x <= event.x <= x + w) and (y <= event.y <= y + h):
                try:
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        SELECT customer_id, name, face_encoding, exit_time,
                               (SELECT COUNT(*) FROM customers c2 
                                WHERE c2.name = c1.name) as total_visits
                        FROM customers c1
                        WHERE customer_id IN (
                            SELECT MAX(customer_id) 
                            FROM customers 
                            GROUP BY name
                        )
                    """)
                    customers = cursor.fetchall()
                    
                    is_known = False
                    for customer_id, name, stored_features, exit_time, total_visits in customers:
                        if stored_features is not None:
                            stored_features = np.frombuffer(stored_features, dtype=np.float64)
                            if self.compare_features(face_data['features'], stored_features):
                                is_known = True
                                if exit_time is not None:
                                    cursor.execute("""
                                        SELECT customer_id FROM customers 
                                        WHERE name = ? AND exit_time IS NULL
                                    """, (name,))
                                    active_entry = cursor.fetchone()
                                    
                                    if not active_entry:
                                        cursor.execute("""
                                            INSERT INTO customers 
                                            (name, face_encoding, entry_time, visit_count)
                                            VALUES (?, ?, datetime('now'), ?)
                                        """, (name, stored_features, total_visits + 1))
                                        self.conn.commit()
                                        messagebox.showinfo("Welcome Back", 
                                            f"Welcome back {name}!\nVisit #{total_visits + 1}")
                                        self.load_existing_customers()
                                    else:
                                        messagebox.showinfo("Info", 
                                            f"{name} is already checked in!")
                                else:
                                    messagebox.showinfo("Info", 
                                        f"{name} is already checked in!")
                                break
                    
                    if not is_known:
                        self.show_registration_dialog(face_data)
                        
                except Exception as e:
                    messagebox.showerror("Error", f"Error processing face: {e}")
                    print(f"Face processing error: {e}")
                break

    def show_registration_dialog(self, face_data):
        """Show registration dialog only for new faces"""
        if self.registration_dialog is not None:
            self.registration_dialog.lift()
            return
            
        dialog = tk.Toplevel(self.root)
        self.registration_dialog = dialog
        dialog.title("Register New Customer")
        dialog.geometry("400x500")
        
        dialog.transient(self.root)
        dialog.grab_set()
        
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')
        
        face_img = cv2.cvtColor(face_data['img'], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_img)
        img = img.resize((250, 250), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        ttk.Label(dialog, text="New Customer Registration", 
                 font=('Helvetica', 14, 'bold')).pack(pady=10)
        
        preview_label = ttk.Label(dialog, image=photo)
        preview_label.image = photo
        preview_label.pack(pady=10)
        
        ttk.Label(dialog, text="Enter Customer Name:").pack(pady=5)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=name_var)
        name_entry.pack(pady=5)
        name_entry.focus()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def register():
            name = name_var.get().strip()
            if name:
                self.register_face(face_data, name)
                dialog.destroy()
                self.registration_dialog = None
            else:
                messagebox.showwarning("Warning", "Please enter a name!")
        
        def close_dialog():
            self.registration_dialog = None
            dialog.destroy()
        
        ttk.Button(
            button_frame,
            text="Register",
            command=register
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=close_dialog
        ).pack(side=tk.LEFT, padx=5)
        
        dialog.bind('<Return>', lambda e: register())
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    def exit_customer(self):
        """Mark selected customer as exited and record purchase details"""
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a customer to mark as exited")
            return
        
        customer_id = self.tree.item(selected_item[0])['values'][0]
        customer_name = self.tree.item(selected_item[0])['values'][1]
        status = self.tree.item(selected_item[0])['values'][4]
        
        if status == "Left":
            messagebox.showinfo("Info", f"{customer_name} has already been marked as exited")
            return
        
        # Show purchase dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Exit Details for {customer_name}")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')
        
        form_frame = ttk.LabelFrame(dialog, text="Purchase Information", padding=10)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Purchase Yes/No
        ttk.Label(form_frame, text="Did the customer make a purchase?").grid(row=0, column=0, columnspan=2, pady=5)
        purchase_var = tk.StringVar(value="No")
        ttk.Radiobutton(form_frame, text="Yes", variable=purchase_var, value="Yes").grid(row=1, column=0, pady=5)
        ttk.Radiobutton(form_frame, text="No", variable=purchase_var, value="No").grid(row=1, column=1, pady=5)
        
        # Purchase details (initially hidden)
        purchase_frame = ttk.Frame(form_frame)
        purchase_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky='ew')
        
        ttk.Label(purchase_frame, text="Product ID:").grid(row=0, column=0, sticky='e', pady=5)
        product_id_var = tk.StringVar()
        ttk.Entry(purchase_frame, textvariable=product_id_var).grid(row=0, column=1, sticky='ew', pady=5)
        
        ttk.Label(purchase_frame, text="Product Name:").grid(row=1, column=0, sticky='e', pady=5)
        product_name_var = tk.StringVar()
        ttk.Entry(purchase_frame, textvariable=product_name_var).grid(row=1, column=1, sticky='ew', pady=5)
        
        ttk.Label(purchase_frame, text="Product Price:").grid(row=2, column=0, sticky='e', pady=5)
        product_price_var = tk.StringVar()
        ttk.Entry(purchase_frame, textvariable=product_price_var).grid(row=2, column=1, sticky='ew', pady=5)
        
        def toggle_purchase_details():
            if purchase_var.get() == "Yes":
                purchase_frame.grid()
            else:
                purchase_frame.grid_remove()
        
        purchase_var.trace('w', lambda *args: toggle_purchase_details())
        toggle_purchase_details()  # Initial state
        
        def submit_exit():
            try:
                cursor = self.conn.cursor()
                
                # Update exit time
                cursor.execute("""
                    UPDATE customers 
                    SET exit_time = datetime('now')
                    WHERE customer_id = ? AND exit_time IS NULL
                """, (customer_id,))
                
                if purchase_var.get() == "Yes":
                    # Validate purchase details
                    product_id = product_id_var.get().strip()
                    product_name = product_name_var.get().strip()
                    product_price = product_price_var.get().strip()
                    
                    if not all([product_id, product_name, product_price]):
                        messagebox.showwarning("Warning", "Please fill all purchase details")
                        return
                        
                    try:
                        product_price = float(product_price)
                    except ValueError:
                        messagebox.showwarning("Warning", "Price must be a number")
                        return
                        
                    # Insert purchase record
                    cursor.execute("""
                        INSERT INTO purchases 
                        (customer_id, product_id, product_name, product_price, purchase_time)
                        VALUES (?, ?, ?, ?, datetime('now'))
                    """, (customer_id, product_id, product_name, product_price))
                
                self.conn.commit()
                
                if cursor.rowcount > 0:
                    messagebox.showinfo("Success", f"Successfully marked {customer_name} as exited")
                    dialog.destroy()
                    self.load_existing_customers()
                else:
                    messagebox.showwarning("Warning", "Customer record not found or already exited")
                    dialog.destroy()
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process exit: {e}")
                print(f"Exit error: {e}")
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Submit", command=submit_exit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def show_past_records(self):
        """Show past records for selected customer including purchase details"""
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a customer to view past records")
            return
        
        customer_name = self.tree.item(selected_item[0])['values'][1]
        
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title(f"Past Records for {customer_name}")
            dialog.geometry("800x500")
            dialog.transient(self.root)
            dialog.grab_set()
            
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            x = (dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (dialog.winfo_screenheight() // 2) - (height // 2)
            dialog.geometry(f'+{x}+{y}')
            
            records_frame = ttk.LabelFrame(dialog, text=f"Visit History for {customer_name}", padding=10)
            records_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            columns = ('Visit #', 'Entry Time', 'Exit Time', 'Duration', 'Product ID', 'Product', 'Price')
            records_tree = ttk.Treeview(records_frame, columns=columns, show='headings')
            
            records_tree.heading('Visit #', text='Visit #')
            records_tree.heading('Entry Time', text='Entry Time')
            records_tree.heading('Exit Time', text='Exit Time')
            records_tree.heading('Duration', text='Duration')
            records_tree.heading('Product ID', text='Product ID')
            records_tree.heading('Product', text='Product')
            records_tree.heading('Price', text='Price ($)')
            
            records_tree.column('Visit #', width=60)
            records_tree.column('Entry Time', width=150)
            records_tree.column('Exit Time', width=150)
            records_tree.column('Duration', width=100)
            records_tree.column('Product ID', width=80)
            records_tree.column('Product', width=120)
            records_tree.column('Price', width=80)
            
            scrollbar = ttk.Scrollbar(records_frame, orient=tk.VERTICAL, command=records_tree.yview)
            records_tree.configure(yscrollcommand=scrollbar.set)
            
            records_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT 
                    c.customer_id, 
                    c.entry_time, 
                    c.exit_time, 
                    c.visit_count,
                    p.product_id,
                    p.product_name,
                    p.product_price
                FROM customers c
                LEFT JOIN purchases p ON c.customer_id = p.customer_id
                WHERE c.name = ?
                ORDER BY c.entry_time DESC
            """, (customer_name,))
            
            records = cursor.fetchall()
            
            for i, (customer_id, entry_time, exit_time, visit_count, product_id, product_name, product_price) in enumerate(records, 1):
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
                    
                    # Handle purchase details explicitly
                    if product_id is None and product_name is None and product_price is None:
                        # No purchase made for this visit
                        product_id_display = "-"
                        product_name_display = "-"
                        product_price_display = "-"
                    else:
                        # Purchase made, use actual values
                        product_id_display = product_id if product_id else "-"
                        product_name_display = product_name if product_name else "-"
                        product_price_display = f"${product_price:.2f}" if product_price is not None else "-"
                    
                    records_tree.insert('', 'end', values=(
                        visit_count,
                        entry_str,
                        exit_str,
                        duration_str,
                        product_id_display,
                        product_name_display,
                        product_price_display
                    ))
                except Exception as e:
                    print(f"Error processing record {customer_id}: {e}")
                    continue
            
            ttk.Button(
                dialog,
                text="Close",
                command=dialog.destroy
            ).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show past records: {e}")
            print(f"Past records error: {e}")

def main():
    root = tk.Tk()
    app = JewelryShopDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()