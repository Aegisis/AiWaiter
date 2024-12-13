import pandas as pd
import cv2
import face_recognition
import threading
import queue
from PIL import Image
from customtkinter import (
    CTk, CTkFrame, CTkLabel, CTkButton, CTkEntry, 
    CTkImage, CTkScrollableFrame
)
from CTkMessagebox import CTkMessagebox
import numpy as np
from backend import Backend
from tkinter import StringVar
from customtkinter import CTkToplevel
from datetime import datetime
import base64
import json
import os
import time

class CameraThread(threading.Thread):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.running = True
        self.last_frame = None
        self.face_detected_frames = 0
        
    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.last_frame = frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    self.face_detected_frames += 1
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    if self.face_detected_frames >= 3:
                        face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]])
                        if face_encodings:
                            self.process_face(frame, face_encodings[0])
                            break
                
                if hasattr(self.parent, 'camera_label'):
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    photo = CTkImage(light_image=image, size=(640, 480))
                    self.parent.camera_label.configure(image=photo)
                    
        cap.release()

    def process_face(self, frame, face_encoding):
        """Process detected face and find best match"""
        try:
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = "user_detected.jpg"
            cv2.imwrite(image_path, frame)
            
            # Convert and save to JSON
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Save face data
            face_data = {
                "timestamp": timestamp,
                "image": img_base64,
                "capture_date": datetime.now().isoformat(),
                "encoding": face_encoding.tolist()
            }
            
            self.parent.save_face_data(face_data)
            
            # Try to find matching user
            user_id, name = self.parent.backend.verify_face(face_encoding)
            
            def show_result():
                if user_id:
                    # If face recognized, go directly to menu
                    self.parent.after(100, lambda: self.parent.show_menu_for_user(user_id, name))
                else:
                    # If face not recognized, show registration choice with captured face data
                    self.parent.after(100, lambda: self.parent.show_registration_choice(face_encoding))
            
            # Schedule the result display
            self.parent.after(0, show_result)
            
        except Exception as e:
            print(f"Error processing face: {e}")

class SmartFoodSystem(CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize file paths
        self.MENU_DATA_PATH = "indian_food.csv"
        self.USER_PHOTOS_PATH = "user_photos.json"
        self.USER_DETECTED_PATH = "user_detected.jpg"
        
        # Initialize backend first
        self.backend = Backend()
        
        # Window configuration
        self.title("GP Robo - Your AI Waiter")
        self.geometry("1200x800")
        self.configure(fg_color="#FFFFFF")  # Light background
        
        # Initialize variables
        self.scan_window = None
        self.camera_thread = None
        self.cart_items = []
        self.cart_window = None
        
        # Load menu data
        self.load_menu_data()
        
        # Setup styles
        self.setup_styles()
        
        # Show landing page first
        self.show_landing_page()

    def setup_styles(self):
        """Setup custom styles with light fonts on light theme"""
        self.styles = {
            'heading': {
                'font': ('Helvetica', 24, 'bold'),
                'text_color': '#000000'  # Black text
            },
            'subheading': {
                'font': ('Helvetica', 18, 'bold'),
                'text_color': '#333333'  # Dark gray text
            },
            'body': {
                'font': ('Helvetica', 12),
                'text_color': '#000000'  # Black text
            }
        }
        
    def show_landing_page(self):
        """Show the landing page with touch to continue"""
        # Main container
        self.main_container = CTkFrame(
            self,
            fg_color="#FFFFFF",  # Light background
            corner_radius=15
        )
        self.main_container.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )
        
        # Welcome text
        CTkLabel(
            self.main_container,
            text="Welcome to GP Robo",
            font=self.styles['heading']['font'],
            text_color='#000000'  # Black text
        ).pack(pady=(50, 10))
        
        CTkLabel(
            self.main_container,
            text="Your AI Waiter",
            font=self.styles['subheading']['font'],
            text_color='#333333'  # Dark gray text
        ).pack(pady=10)
        
        # Robot icon
        CTkLabel(
            self.main_container,
            text="🤖",
            font=("Helvetica", 72),
            text_color='#000000'  # Black text
        ).pack(pady=30)
        
        # Touch to continue button
        touch_button = CTkButton(
            self.main_container,
            text="Touch to Continue",
            font=self.styles['heading']['font'],
            fg_color="#4CAF50",
            hover_color="#45a049",
            width=300,
            height=100,
            corner_radius=50,
            command=self.show_options_page
        )
        touch_button.pack(pady=50)

    def show_options_page(self):
        """Show face detection directly after touch to continue"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Start face detection immediately
        self.start_face_recognition()

    def show_registration_choice(self, face_encoding):
        """Show stylish registration choice with modern UI"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Main content frame with gradient background
        content_frame = CTkFrame(
            self.main_container,
            fg_color=("#f0f2f5", "#2b2b2b"),  # Light/Dark mode colors
            corner_radius=20
        )
        content_frame.pack(expand=True, fill="both", padx=40, pady=40)
        
        # Welcome message
        CTkLabel(
            content_frame,
            text="Welcome to GP Robo! 🤖",
            font=("Helvetica", 32, "bold"),
            text_color=("#1a1a1a", "#ffffff")
        ).pack(pady=(40, 20))
        
        # Subtitle
        CTkLabel(
            content_frame,
            text="Your AI-Powered Dining Experience",
            font=("Helvetica", 16),
            text_color=("#666666", "#aaaaaa")
        ).pack(pady=(0, 30))
        
        # Benefits frame with modern card design
        benefits_frame = CTkFrame(
            content_frame,
            fg_color=("#ffffff", "#333333"),
            corner_radius=15,
            border_width=1,
            border_color=("#e0e0e0", "#404040")
        )
        benefits_frame.pack(padx=60, pady=(0, 30), fill="x")
        
        # Benefits content with icons
        benefits = [
            ("🎁", "Exclusive Rewards", "Special birthday treats and member discounts"),
            ("🎯", "Personalized Menu", "AI-powered food recommendations"),
            ("⚡", "Quick Access", "Instant recognition for faster ordering"),
            ("💝", "Member Perks", "Special offers and early access to new items")
        ]
        
        for icon, title, desc in benefits:
            benefit_item = CTkFrame(
                benefits_frame,
                fg_color="transparent"
            )
            benefit_item.pack(padx=20, pady=10, fill="x")
            
            # Icon
            CTkLabel(
                benefit_item,
                text=icon,
                font=("Helvetica", 24)
            ).pack(side="left", padx=(10, 15))
            
            # Benefit text frame
            text_frame = CTkFrame(
                benefit_item,
                fg_color="transparent"
            )
            text_frame.pack(side="left", fill="x", expand=True)
            
            CTkLabel(
                text_frame,
                text=title,
                font=("Helvetica", 14, "bold"),
                anchor="w"
            ).pack(fill="x")
            
            CTkLabel(
                text_frame,
                text=desc,
                font=("Helvetica", 12),
                text_color=("#666666", "#aaaaaa"),
                anchor="w"
            ).pack(fill="x")
        
        # Buttons frame with modern design
        buttons_frame = CTkFrame(
            content_frame,
            fg_color="transparent"
        )
        buttons_frame.pack(pady=30)
        
        # Register button with gradient effect
        register_btn = CTkButton(
            buttons_frame,
            text="Register Now",
            font=("Helvetica", 16, "bold"),
            fg_color=("#FF3C5A", "#FF1A1A"),
            hover_color=("#FF1A1A", "#FF0000"),
            corner_radius=25,
            width=200,
            height=50,
            command=lambda: self.show_registration_form(face_encoding)
        )
        register_btn.pack(side="left", padx=10)
        
        # Guest button with outlined style
        guest_btn = CTkButton(
            buttons_frame,
            text="Continue as Guest",
            font=("Helvetica", 16, "bold"),
            fg_color="transparent",
            hover_color=("#e0e0e0", "#404040"),
            corner_radius=25,
            width=200,
            height=50,
            border_width=2,
            border_color=("#666666", "#aaaaaa"),
            command=self.show_menu_without_registration
        )
        guest_btn.pack(side="left", padx=10)
        
        # Privacy note
        CTkLabel(
            content_frame,
            text="🔒 Your data is protected and never shared",
            font=("Helvetica", 12),
            text_color=("#666666", "#aaaaaa")
        ).pack(pady=(20, 40))

    def start_face_recognition(self, require_smile=False):
        """Start face recognition with automatic capture"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Create animation frame
        animation_frame = CTkFrame(self.main_container)
        animation_frame.pack(expand=True)
        
        # Initialize camera thread first
        self.camera_thread = CameraThread(self)
        self.camera_thread.daemon = True
        
        # Create animation label as instance variable
        self.animation_label = CTkLabel(animation_frame, text="")
        self.animation_label.pack(expand=True)
        
        try:
            # Load animation
            self.animation = cv2.VideoCapture("Animation - 1733991071989.mp4")
            if not self.animation.isOpened():
                raise Exception("Could not open animation file")
            
            # Get video dimensions
            width = int(self.animation.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.animation.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            def update_animation():
                if not hasattr(self, 'animation') or not self.animation.isOpened():
                    return
                    
                try:
                    ret, frame = self.animation.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)
                        photo = CTkImage(light_image=image, size=(width, height))
                        if hasattr(self, 'animation_label') and self.animation_label.winfo_exists():
                            self.animation_label.configure(image=photo)
                            self.animation_label.image = photo  # Keep reference
                    else:
                        self.animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                    if hasattr(self, 'animation'):
                        self.after(33, update_animation)
                except Exception as e:
                    print(f"Animation error: {e}")
            
            # Start animation
            update_animation()
            
            # Start timeout timer
            self.face_detection_start = time.time()
            def check_timeout():
                if not hasattr(self, 'camera_thread'):
                    return
                    
                elapsed_time = time.time() - self.face_detection_start
                if elapsed_time > 6:  # 6 seconds timeout
                    if hasattr(self, 'animation'):
                        self.animation.release()
                    if self.camera_thread and self.camera_thread.is_alive():
                        self.camera_thread.running = False
                        self.camera_thread.join()
                    self.show_registration_choice(None)
                else:
                    self.after(100000, check_timeout)
            
            # Start timeout checker
            check_timeout()
            
        except Exception as e:
            print(f"Error loading animation: {e}")
            CTkLabel(
                animation_frame,
                text="🤖 Scanning...",
                font=("Helvetica", 24)
            ).pack(pady=20)
        
        # Status messages
        self.status_label = CTkLabel(
            self.main_container,
            text="Detecting face...",
            font=self.styles['body']['font']
        )
        self.status_label.pack(pady=10)
        
        def update_status():
            if not hasattr(self, 'status_label'):
                return
                
            try:
                current_text = self.status_label.cget("text")
                if "Detecting face" in current_text:
                    self.status_label.configure(text="Scanning face...")
                elif "Scanning face" in current_text:
                    self.status_label.configure(text="Processing...")
                elif "Processing" in current_text:
                    self.status_label.configure(text="Detecting face...")
                
                if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
                    self.after(1000, update_status)
            except Exception as e:
                print(f"Error updating status: {e}")
        
        # Start status updates
        update_status()
        
        # Cancel button
        CTkButton(
            self.main_container,
            text="Cancel",
            font=self.styles['body']['font'],
            command=self.stop_camera_and_return,
            fg_color="#f44336",
            hover_color="#da190b",
            width=150
        ).pack(pady=10)
        
        # Start camera thread
        self.camera_thread.start()

    def capture_photo(self):
        """Capture photo and process face recognition"""
        if hasattr(self, 'camera_thread') and self.camera_thread.last_frame is not None:
            try:
                frame = self.camera_thread.last_frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if not face_locations:
                    CTkMessagebox(title="Error", message="No face detected! Please try again.")
                    return
                
                # Get face encodings first
                face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]])
                if not face_encodings:
                    CTkMessagebox(title="Error", message="Could not process face. Please try again.")
                    return
                
                # Create photos directory if it doesn't exist
                photos_dir = os.path.join(os.getcwd(), "user_data", "photos")
                os.makedirs(photos_dir, exist_ok=True)
                
                # Save image with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(photos_dir, f"user_{timestamp}.jpg")
                
                # Save physical image
                cv2.imwrite(image_path, frame)
                
                # Convert image to base64 for JSON storage
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Save to JSON
                self.save_image_to_json(img_base64, timestamp)
                
                # Clear camera display
                self.camera_label.configure(image=None)
                
                # Check if user exists
                user_id, name = self.backend.verify_face(face_encodings[0])
                
                if user_id:
                    CTkMessagebox(title="Success", message=f"Welcome back, {name}!")
                    self.show_menu_for_user(user_id, name)
                else:
                    CTkMessagebox(title="New User", message="Face not recognized. Please register.")
                    self.show_registration_form(face_encodings[0])
            except Exception as e:
                print(f"Error capturing photo: {e}")
                CTkMessagebox(title="Error", message="Failed to capture photo. Please try again.")

    def save_image_to_json(self, img_base64, timestamp):
        """Save captured image to JSON database"""
        json_file = "user_photos.json"
        
        try:
            # Load existing data
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {"images": []}
            
            # Add new image
            image_data = {
                "timestamp": timestamp,
                "image": img_base64,
                "capture_date": datetime.now().isoformat()
            }
            
            data["images"].append(image_data)
            
            # Save updated data
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
                
        except Exception as e:
            print(f"Error saving image to JSON: {e}")

    def stop_camera_and_show_menu(self, user_id, name):
        """Stop camera and show user menu"""
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.running = False
            self.camera_thread.join()
        self.show_menu_for_user(user_id, name)

    def stop_camera_and_register(self, face_encoding):
        """Stop camera and show registration form"""
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.running = False
            self.camera_thread.join()
        self.show_registration_form(face_encoding)

    def stop_camera_and_return(self):
        """Stop camera and return to landing page"""
        if hasattr(self, 'animation_running'):
            self.animation_running = False
            
        if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.running = False
            self.camera_thread.join()
            
        self.show_landing_page()

    def load_menu_data(self):
        """Load menu data from CSV file"""
        try:
            self.menu_data = pd.read_csv(self.MENU_DATA_PATH)
            # Clean up course names to match our categories
            self.menu_data['Course'] = self.menu_data['Course'].map({
                'dessert': 'Dessert',
                'main course': 'Main Course',
                'snack': 'Snacks',
                'snacks': 'Snacks'
            }).fillna('Other')
            print(f"Loaded {len(self.menu_data)} menu items")
        except Exception as e:
            print(f"Error loading menu data: {e}")
            self.menu_data = pd.DataFrame()

    def show_error(self, message):
        """Display error message to user."""
        CTkMessagebox(
            title="Error",
            message=message,
            icon="cancel"
        )

    def show_registration_form(self, face_encoding):
        """Show registration form with validation"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Registration form container
        form_frame = CTkFrame(
            self.main_container,
            fg_color="#333333",
            corner_radius=15
        )
        form_frame.pack(expand=True, padx=20, pady=20)
        
        # Title
        CTkLabel(
            form_frame,
            text="New User Registration",
            font=self.styles['heading']['font']
        ).pack(pady=20)
        
        # Form fields with validation
        fields = {
            'name': {
                'label': "Full Name",
                'validate': lambda x: len(x.strip()) >= 3,
                'error': "Name must be at least 3 characters"
            },
            'phone': {
                'label': "Phone Number",
                'validate': lambda x: x.isdigit() and len(x) == 10,
                'error': "Phone number must be 10 digits"
            },
            'dob': {
                'label': "Date of Birth (YYYY-MM-DD)",
                'validate': lambda x: self.validate_date(x),
                'error': "Invalid date format. Use YYYY-MM-DD"
            }
        }
        
        entries = {}
        error_labels = {}
        
        for field, config in fields.items():
            # Field container
            field_frame = CTkFrame(form_frame, fg_color="transparent")
            field_frame.pack(pady=5)
            
            # Field label
            CTkLabel(
                field_frame,
                text=config['label'],
                font=self.styles['body']['font']
            ).pack(anchor="w")
            
            # Entry field
            entries[field] = CTkEntry(
                field_frame,
                width=300,
                placeholder_text=config['label']
            )
            entries[field].pack(pady=(5,0))
            
            # Error label (hidden by default)
            error_labels[field] = CTkLabel(
                field_frame,
                text=config['error'],
                text_color="#FF3333",
                font=('Helvetica', 10)
            )
            
            def create_validator(field_name):
                def validate(event):
                    value = entries[field_name].get()
                    if not fields[field_name]['validate'](value):
                        error_labels[field_name].pack(pady=(0,5))
                        entries[field_name].configure(text_color="#FF3333")
                    else:
                        error_labels[field_name].pack_forget()
                        entries[field_name].configure(text_color=self.styles['body']['text_color'])
                return validate
            
            # Add validation on key release
            entries[field].bind('<KeyRelease>', create_validator(field))
        
        def submit_registration():
            # Validate all fields
            valid = True
            for field, config in fields.items():
                if not config['validate'](entries[field].get()):
                    error_labels[field].pack(pady=(0,5))
                    entries[field].configure(text_color="#FF3333")
                    valid = False
                else:
                    error_labels[field].pack_forget()
                    entries[field].configure(text_color=self.styles['body']['text_color'])
            
            if not valid:
                return
            
            try:
                # Clean input data
                name = entries['name'].get().strip()
                phone = entries['phone'].get().strip()
                dob = entries['dob'].get().strip()
                
                print(f"Attempting to register user: {name}")
                
                # Register user in database
                user_id = self.backend.register_user(
                    name,
                    phone,
                    dob,
                    face_encoding
                )
                
                if user_id:
                    print(f"Registration successful. User ID: {user_id}")
                    CTkMessagebox(
                        title="Success",
                        message="Registration successful!",
                        icon="check"
                    )
                    # Show menu for new user
                    self.show_menu_for_user(user_id, name)
                else:
                    print("Registration failed")
                    self.show_error("Registration failed!")
                    
            except Exception as e:
                print(f"Registration error: {str(e)}")
                self.show_error(f"Registration error: {str(e)}")
        
        # Submit button
        CTkButton(
            form_frame,
            text="Register",
            font=self.styles['body']['font'],
            fg_color="#28a745",
            hover_color="#218838",
            command=submit_registration
        ).pack(pady=20)
        
        # Cancel button
        CTkButton(
            form_frame,
            text="Cancel",
            font=self.styles['body']['font'],
            fg_color="#dc3545",
            hover_color="#c82333",
            command=self.show_landing_page
        ).pack(pady=(0,20))

    def validate_field(self, entry, error_label, validator, error_text):
        """Validate a single form field"""
        value = entry.get()
        if not validator(value):
            error_label.pack(pady=(0,5))
            entry.configure(border_color="#FF3333")  # Red border for invalid
            return False
        else:
            error_label.pack_forget()
            entry.configure(border_color="transparent")  # Reset border
            return True

    def validate_date(self, date_str):
        """Validate date format and range"""
        try:
            # Parse the date
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Check if date is not in future
            if date > datetime.now():
                return False
            
            # Check if age is reasonable (e.g., > 5 years)
            age = (datetime.now() - date).days / 365
            return age >= 5
            
        except ValueError:
            return False

    def show_menu_for_user(self, user_id, name):
        """Show menu for registered user with categories and cart"""
        try:
            # Clear main container
            for widget in self.main_container.winfo_children():
                widget.destroy()
            
            # Create header with user info and cart
            header_frame = CTkFrame(self.main_container, fg_color="#2D2D2D")
            header_frame.pack(fill="x", padx=20, pady=10)
            
            CTkLabel(
                header_frame,
                text=f"Welcome, {name}!",
                font=self.styles['subheading']['font']
            ).pack(side="left", padx=20)
            
            # Profile and Cart buttons
            CTkButton(
                header_frame,
                text="👤 Profile",
                command=lambda: self.show_user_profile(user_id),
                fg_color="#FF3C5A",
                hover_color="#FF1A1A"
            ).pack(side="right", padx=5)
            
            # Main content area with left sidebar and right content
            content_frame = CTkFrame(self.main_container, fg_color="#1A1A1A")
            content_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            # Left sidebar with cart and recommendations
            left_sidebar = CTkFrame(content_frame, fg_color="#2D2D2D", width=300)
            left_sidebar.pack(side="left", fill="y", padx=(0, 20), pady=10)
            
            # Top frame for cart
            cart_frame = CTkFrame(left_sidebar, fg_color="#333333")
            cart_frame.pack(fill="x", padx=10, pady=10)
            
            CTkLabel(
                cart_frame,
                text="Your Order",
                font=self.styles['subheading']['font']
            ).pack(pady=10)
            
            # Scrollable cart items
            cart_items_frame = CTkScrollableFrame(cart_frame, height=200)
            cart_items_frame.pack(fill="x", padx=10, pady=5)
            
            total_amount = 0
            for item in self.cart_items:
                item_frame = CTkFrame(cart_items_frame, fg_color="#404040")
                item_frame.pack(fill="x", pady=2)
                
                CTkLabel(
                    item_frame,
                    text=item['Name'],
                    font=self.styles['body']['font']
                ).pack(side="left", padx=5)
                
                CTkLabel(
                    item_frame,
                    text=f"₹{item.get('Price', 100)}",
                    font=self.styles['body']['font']
                ).pack(side="right", padx=5)
                
                total_amount += item.get('Price', 100)
            
            # Total and Order button
            CTkLabel(
                cart_frame,
                text=f"Total: ₹{total_amount}",
                font=self.styles['subheading']['font']
            ).pack(pady=10)
            
            CTkButton(
                cart_frame,
                text="Order Now",
                command=lambda: self.show_cart(user_id),
                fg_color="#28a745",
                hover_color="#218838"
            ).pack(pady=10)
            
            # Bottom frame for recommendations
            CTkLabel(
                left_sidebar,
                text="Recommended for You",
                font=self.styles['subheading']['font']
            ).pack(pady=10)
            
            rec_frame = CTkScrollableFrame(left_sidebar, height=300)
            rec_frame.pack(fill="both", expand=True, padx=10, pady=5)
            
            # Get recommendations safely
            try:
                recommendations = self.backend.get_recommendations(user_id) or []
                
                # Show recommendations in the bottom frame
                for item_name in recommendations[:5]:
                    item_frame = CTkFrame(rec_frame, fg_color="#404040")
                    item_frame.pack(fill="x", pady=2)
                    
                    CTkLabel(
                        item_frame,
                        text=item_name,
                        font=self.styles['body']['font']
                    ).pack(side="left", padx=5)
                    
                    CTkButton(
                        item_frame,
                        text="Add",
                        command=lambda i=self.menu_data[self.menu_data['Name'] == item_name].iloc[0]: self.add_to_cart(i, user_id),
                        fg_color="#28a745",
                        width=60,
                        height=24
                    ).pack(side="right", padx=5)
                    
            except Exception as e:
                print(f"Error displaying recommendations: {e}")
            
            # Right side menu categories
            menu_frame = CTkFrame(content_frame, fg_color="#262626")
            menu_frame.pack(side="right", fill="both", expand=True)
            
            # Category buttons
            categories_frame = CTkFrame(menu_frame, fg_color="transparent")
            categories_frame.pack(fill="x", padx=20, pady=10)
            
            categories = {
                "Main Course": "🍛",
                "Dessert": "🍨",
                "Snacks": "🥪"
            }
            
            for category, emoji in categories.items():
                CTkButton(
                    categories_frame,
                    text=f"{emoji} {category}",
                    command=lambda c=category: self.show_category(c, user_id),
                    fg_color="#333333",
                    hover_color="#404040"
                ).pack(side="left", padx=5)
            
            # Menu items frame
            self.menu_frame = CTkScrollableFrame(menu_frame, fg_color="transparent")
            self.menu_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            # Show first category by default
            self.show_category("Main Course", user_id)
            
        except Exception as e:
            print(f"Error showing menu: {e}")
            CTkMessagebox(
                title="Error",
                message="An error occurred while loading the menu. Please try again.",
                icon="cancel"
            )
            self.show_landing_page()

    def show_user_profile(self, user_id):
        """Show user profile in a separate window"""
        try:
            # Get user details from database
            user_details = self.backend.get_user_details(user_id)
            
            # Create new window with dark theme
            profile_window = CTkToplevel()
            profile_window.grab_set()  # Make window modal
            profile_window.title("User Profile")
            profile_window.geometry("400x600")
            profile_window.configure(fg_color="#1A1A1A")
            
            # Center the window
            profile_window.update_idletasks()
            x = (profile_window.winfo_screenwidth() - profile_window.winfo_width()) // 2
            y = (profile_window.winfo_screenheight() - profile_window.winfo_height()) // 2
            profile_window.geometry(f"+{x}+{y}")
            
            # Main profile frame
            profile_frame = CTkFrame(
                profile_window,
                fg_color="#2D2D2D",  # Darker gray
                corner_radius=15,
            )
            profile_frame.pack(expand=True, fill="both", padx=20, pady=20)
            
            # Title
            CTkLabel(
                profile_frame,
                text="User Profile",
                font=self.styles['heading']['font']
            ).pack(pady=20)
            
            try:
                # Load and display user image
                user_image = Image.open("D:/ai project gp robo/aimlproject-main/user_detected.jpg")
                user_image = user_image.resize((150, 150), Image.LANCZOS)
                photo = CTkImage(light_image=user_image, size=(150, 150))
                
                image_label = CTkLabel(
                    profile_frame,
                    image=photo,
                    text=""
                )
                image_label.pack(pady=10)
            except Exception as e:
                print(f"Error loading user image: {e}")
            
            # User details
            details_frame = CTkFrame(profile_frame, fg_color="transparent")
            details_frame.pack(pady=20, padx=30, fill="x")
            
            # Name
            CTkLabel(
                details_frame,
                text="Name:",
                font=self.styles['body']['font']
            ).pack(anchor="w")
            CTkLabel(
                details_frame,
                text=user_details['name'],
                font=('Helvetica', 14, 'bold')
            ).pack(anchor="w", pady=(0, 10))
            
            # Phone
            CTkLabel(
                details_frame,
                text="Phone:",
                font=self.styles['body']['font']
            ).pack(anchor="w")
            CTkLabel(
                details_frame,
                text=user_details['phone'],
                font=('Helvetica', 14, 'bold')
            ).pack(anchor="w", pady=(0, 10))
            
            # DOB
            CTkLabel(
                details_frame,
                text="Date of Birth:",
                font=self.styles['body']['font']
            ).pack(anchor="w")
            CTkLabel(
                details_frame,
                text=user_details['dob'],
                font=('Helvetica', 14, 'bold')
            ).pack(anchor="w", pady=(0, 10))
            
            # Buttons frame
            buttons_frame = CTkFrame(profile_frame, fg_color="transparent")
            buttons_frame.pack(pady=20)
            
            def close_profile():
                profile_window.grab_release()
                profile_window.destroy()
            
            # Update button commands
            CTkButton(
                buttons_frame,
                text="Close",
                command=close_profile,
                fg_color="#FF3C5A",
                hover_color="#FF1A1A",
                width=120
            ).pack(side="left", padx=10)
            
            CTkButton(
                buttons_frame,
                text="Cancel",
                command=close_profile,
                fg_color="#666666",
                hover_color="#555555",
                width=120
            ).pack(side="left", padx=10)
            
        except Exception as e:
            print(f"Error showing profile: {e}")
            CTkMessagebox(title="Error", message="Could not load profile")

    def add_to_cart(self, item, user_id=None):
        """Add item to cart"""
        try:
            # Add price to item dictionary
            item_with_price = item.copy()
            item_with_price['Price'] = 100  # Replace with actual price
            
            self.cart_items.append(item_with_price)
            print(f"Added to cart: {item['Name']}")
            print(f"Cart now has {len(self.cart_items)} items")
            
            CTkMessagebox(
                title="Added to Cart",
                message=f"{item['Name']} added to cart!\nPrice: ₹{item_with_price['Price']}",
                icon="check"
            )
            
            # Refresh the current view - use the correct method based on user_id
            if user_id:
                self.show_menu_for_user(user_id, self.backend.get_user_name(user_id))
            else:
                self.show_menu_without_registration()
                
        except Exception as e:
            print(f"Error adding to cart: {e}")
            CTkMessagebox(
                title="Error",
                message="Could not add item to cart",
                icon="cancel"
            )

    def show_cart(self, user_id):
        """Show cart with order confirmation"""
        if not self.cart_items:
            CTkMessagebox(title="Cart Empty", message="Your cart is empty!")
            return
        
        try:
            # Create and configure cart window
            cart_window = CTkToplevel(self)
            cart_window.title("Your Cart")
            cart_window.geometry("500x600")
            cart_window.configure(fg_color="#1A1A1A")
            
            # Make window modal and wait for it to be ready
            cart_window.transient(self)
            cart_window.focus_force()
            cart_window.grab_set()
            
            # Cart items container
            cart_frame = CTkScrollableFrame(
                cart_window,
                fg_color="#262626"
            )
            cart_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            total_amount = 0
            
            # Display cart items
            for item in self.cart_items:
                item_frame = CTkFrame(cart_frame, fg_color="#333333")
                item_frame.pack(fill="x", padx=10, pady=5)
                
                CTkLabel(
                    item_frame,
                    text=item['Name'],
                    font=self.styles['body']['font'],
                    text_color='#FFFFFF'
                ).pack(side="left", padx=10, pady=10)
                
                price = 100
                total_amount += price
                
                CTkLabel(
                    item_frame,
                    text=f"₹{price}",
                    font=self.styles['body']['font'],
                    text_color='#FFFFFF'
                ).pack(side="right", padx=10)
                
                CTkButton(
                    item_frame,
                    text="Remove",
                    command=lambda i=item: self.remove_from_cart(i, cart_window, user_id),
                    fg_color="#dc3545",
                    hover_color="#c82333",
                    width=80
                ).pack(side="right", padx=5)
            
            # Total amount
            total_frame = CTkFrame(cart_window, fg_color="#333333")
            total_frame.pack(fill="x", padx=20, pady=10)
            
            CTkLabel(
                total_frame,
                text=f"Total: ₹{total_amount}",
                font=self.styles['subheading']['font'],
                text_color='#FFFFFF'
            ).pack(pady=10)
            
            # Confirm order button
            CTkButton(
                cart_window,
                text="Confirm Order",
                command=lambda: self.place_order(user_id, cart_window),
                fg_color="#28a745",
                hover_color="#218838",
                width=200
            ).pack(pady=20)
            
        except Exception as e:
            print(f"Error showing cart: {e}")
            CTkMessagebox(title="Error", message="Could not load cart")

    def place_order(self, user_id, cart_window):
        """Place order and update order history"""
        if not self.cart_items:
            CTkMessagebox(title="Error", message="Cart is empty!")
            return
        
        try:
            # Get order items
            order_items = [item['Name'] for item in self.cart_items]
            
            # Save order to database
            self.backend.save_order(user_id, order_items, total_amount=len(order_items) * 100)
            
            # Clear cart and destroy window safely
            self.cart_items = []
            
            # Release window grab before destroying
            if cart_window.winfo_exists():
                cart_window.grab_release()
                cart_window.destroy()
            
            def cleanup_and_return():
                # Show success and thank you messages
                CTkMessagebox(
                    title="Success",
                    message="Order placed successfully!",
                    icon="check"
                )
                
                CTkMessagebox(
                    title="Thank You",
                    message="Thank you for your order!\nReturning to main page...",
                    icon="check"
                )
                
                # Clear all widgets
                for widget in self.winfo_children():
                    widget.destroy()
                
                # Return to landing page
                self.show_landing_page()
                
                # Optional: Force garbage collection
                import gc
                gc.collect()
            
            # Execute cleanup after a short delay
            self.after(1000, cleanup_and_return)
            
        except Exception as e:
            print(f"Error placing order: {e}")
            CTkMessagebox(title="Error", message="Failed to place order!")

    def remove_from_cart(self, item, cart_window, user_id):
        """Remove item from cart"""
        self.cart_items.remove(item)
        cart_window.destroy()
        self.show_cart(user_id)  # Refresh cart window

    def show_menu_without_registration(self):
        """Show menu for guest users without registration"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Create header
        header_frame = CTkFrame(self.main_container, fg_color="#333333")
        header_frame.pack(fill="x", padx=20, pady=10)
        
        CTkLabel(
            header_frame,
            text="Welcome, Guest!",
            font=self.styles['heading']['font']
        ).pack(side="left", padx=20)
        
        # Cart button for guest
        CTkButton(
            header_frame,
            text=f"🛒 Cart ({len(self.cart_items)})",
            command=lambda: self.show_guest_cart(),
            fg_color="#FF3C5A"
        ).pack(side="right", padx=20)
        
        # Register button
        CTkButton(
            header_frame,
            text="Register for Benefits",
            command=lambda: self.show_consent_form(),
            fg_color="#28a745"
        ).pack(side="right", padx=20)
        
        # Menu container
        menu_frame = CTkScrollableFrame(
            self.main_container,
            fg_color="#262626"
        )
        menu_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Menu items
        for index, row in self.menu_data.iterrows():
            item_frame = CTkFrame(menu_frame, fg_color="#333333")
            item_frame.pack(fill="x", padx=10, pady=5)
            
            # Food details
            details_frame = CTkFrame(item_frame, fg_color="transparent")
            details_frame.pack(side="left", fill="x", expand=True, padx=10, pady=5)
            
            CTkLabel(
                details_frame,
                text=row['Name'],
                font=self.styles['subheading']['font']
            ).pack(anchor="w")
            
            CTkLabel(
                details_frame,
                text=f"Type: {row['Veg_Non']} | Course: {row['Course']}",
                font=self.styles['body']['font']
            ).pack(anchor="w")
            
            # Price and Order button
            button_frame = CTkFrame(item_frame, fg_color="transparent")
            button_frame.pack(side="right", padx=10)
            
            CTkLabel(
                button_frame,
                text="₹100",  # Replace with actual price
                font=self.styles['body']['font']
            ).pack(side="left", padx=10)
            
            CTkButton(
                button_frame,
                text="Add to Cart",
                command=lambda item=row: self.add_to_guest_cart(item),
                fg_color="#28a745"
            ).pack(side="left", padx=5)

    def add_to_guest_cart(self, item):
        """Add item to guest cart"""
        self.cart_items.append(item)
        CTkMessagebox(title="Success", message=f"{item['Name']} added to cart!")
        self.show_menu_without_registration()

    def show_guest_cart(self):
        """Show cart contents for guest"""
        if not self.cart_items:
            CTkMessagebox(title="Cart Empty", message="Your cart is empty!")
            return
        
        cart_window = CTkToplevel(self)
        cart_window.title("Shopping Cart")
        cart_window.geometry("500x600")
        
        # Cart items container
        cart_frame = CTkScrollableFrame(cart_window)
        cart_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        total_amount = 0
        
        # Display cart items
        for item in self.cart_items:
            item_frame = CTkFrame(cart_frame, fg_color="#333333")
            item_frame.pack(fill="x", pady=5)
            
            CTkLabel(
                item_frame,
                text=item['Name'],
                font=self.styles['body']['font']
            ).pack(side="left", padx=10, pady=10)
            
            price = 100  # Replace with actual price
            total_amount += price
            
            CTkLabel(
                item_frame,
                text=f"₹{price}",
                font=self.styles['body']['font']
            ).pack(side="right", padx=10)
            
            CTkButton(
                item_frame,
                text="Remove",
                command=lambda i=item: self.remove_from_guest_cart(i),
                fg_color="#dc3545",
                width=80
            ).pack(side="right", padx=5)
        
        # Total amount
        total_frame = CTkFrame(cart_window, fg_color="#333333")
        total_frame.pack(fill="x", padx=20, pady=10)
        
        CTkLabel(
            total_frame,
            text=f"Total: ₹{total_amount}",
            font=self.styles['subheading']['font']
        ).pack(pady=10)
        
        # Order button
        CTkButton(
            cart_window,
            text="Place Order",
            command=lambda: self.place_guest_order(cart_window),
            fg_color="#28a745"
        ).pack(pady=20)

    def remove_from_guest_cart(self, item):
        """Remove item from guest cart"""
        self.cart_items.remove(item)
        self.show_guest_cart()

    def place_guest_order(self, cart_window):
        """Place order for guest"""
        # Get guest details
        details_window = CTkToplevel(self)
        details_window.title("Order Details")
        details_window.geometry("400x300")
        
        CTkLabel(
            details_window,
            text="Enter Your Details",
            font=self.styles['heading']['font']
        ).pack(pady=20)
        
        # Name field
        CTkLabel(details_window, text="Name:").pack()
        name_var = StringVar()
        CTkEntry(details_window, textvariable=name_var).pack(pady=5)
        
        # Phone field
        CTkLabel(details_window, text="Phone:").pack()
        phone_var = StringVar()
        CTkEntry(details_window, textvariable=phone_var).pack(pady=5)
        
        def confirm_order():
            if not all([name_var.get(), phone_var.get()]):
                CTkMessagebox(title="Error", message="Please fill all fields!")
                return
            
            CTkMessagebox(title="Success", 
                         message=f"Order placed successfully!\nThank you {name_var.get()}!")
            self.cart_items = []  # Clear cart
            details_window.destroy()
            cart_window.destroy()
            self.show_menu_without_registration()
        
        CTkButton(
            details_window,
            text="Confirm Order",
            command=confirm_order,
            fg_color="#28a745"
        ).pack(pady=20)

    def save_face_data(self, face_data):
        """Save face data to JSON database"""
        try:
            # Create base directory if it doesn't exist
            base_dir = os.path.join(os.getcwd(), "user_data")
            os.makedirs(base_dir, exist_ok=True)
            
            # Set paths relative to base directory
            json_file = os.path.join(base_dir, "user_photos.json")
            
            # Load existing data or create new
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {"faces": []}
            
            # Add new face data
            data["faces"].append(face_data)
            
            # Save updated data
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
            
        except Exception as e:
            print(f"Error saving face data: {e}")

    def show_category(self, category, user_id=None):
        """Display menu items for selected category"""
        # Clear current items
        for widget in self.menu_frame.winfo_children():
            widget.destroy()
        
        # Category title
        CTkLabel(
            self.menu_frame,
            text=f"{category} Menu",
            font=self.styles['subheading']['font']
        ).pack(pady=(20,10))
        
        # Filter menu items by category
        category_items = self.menu_data[self.menu_data['Course'] == category]
        
        if category_items.empty:
            CTkLabel(
                self.menu_frame,
                text=f"No items found in {category}",
                font=self.styles['body']['font']
            ).pack(pady=20)
            return
        
        # Create scrollable frame for items
        items_frame = CTkScrollableFrame(self.menu_frame, fg_color="transparent")
        items_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Display items
        for _, item in category_items.iterrows():
            item_frame = CTkFrame(items_frame, fg_color="#333333")
            item_frame.pack(fill="x", padx=10, pady=5)
            
            # Left side - Item details
            details_frame = CTkFrame(item_frame, fg_color="transparent")
            details_frame.pack(side="left", fill="x", expand=True, padx=10, pady=5)
            
            # Item name
            CTkLabel(
                details_frame,
                text=item['Name'],
                font=self.styles['body']['font'],
                text_color="#FFFFFF"
            ).pack(anchor="w")
            
            # Item details
            CTkLabel(
                details_frame,
                text=f"Type: {item['Veg_Non']} | Flavor: {item['Flavour']}",
                font=('Helvetica', 10),
                text_color="#CCCCCC"
            ).pack(anchor="w")
            
            # Right side - Price and Add to Cart
            action_frame = CTkFrame(item_frame, fg_color="transparent")
            action_frame.pack(side="right", padx=10)
            
            # Price
            price = 100  # Replace with actual price from your data if available
            CTkLabel(
                action_frame,
                text=f"₹{price}",
                font=self.styles['body']['font'],
                text_color="#4CAF50"  # Green color for price
            ).pack(side="left", padx=(0,10))
            
            # Add to cart button
            CTkButton(
                action_frame,
                text="🛒 Add",
                command=lambda i=item: self.add_to_cart(i, user_id) if user_id else self.add_to_guest_cart(i),
                fg_color="#28a745",
                hover_color="#218838",
                width=80,
                height=32,
                corner_radius=16
            ).pack(side="left")
            
            # Optional: Add quantity selector
            # quantity_var = StringVar(value="1")
            # CTkEntry(
            #     action_frame,
            #     textvariable=quantity_var,
            #     width=40,
            #     height=32
            # ).pack(side="left", padx=5)

    def show_recommendations(self, user_id):
        """Show recommendations based on past orders and popular items"""
        # Create recommendations frame
        rec_frame = CTkFrame(self.menu_frame, fg_color="#333333")
        rec_frame.pack(fill="x", padx=10, pady=10)
        
        # Title
        CTkLabel(
            rec_frame,
            text="✨ Recommended for You",
            font=self.styles['subheading']['font']
        ).pack(pady=10)
        
        # Get past orders based recommendations
        past_orders = self.backend.get_user_orders(user_id)
        recommended_items = []
        
        if past_orders:
            # Extract past ordered items
            past_items = []
            for _, items, _ in past_orders:
                past_items.extend(items.split(','))
            
            # Get similar items based on past orders
            for item in past_items:
                item = item.strip()
                # Check if item exists in menu data
                matching_items = self.menu_data[self.menu_data['Name'] == item]
                if not matching_items.empty:
                    item_category = matching_items['Course'].iloc[0]
                    # Get items from same category
                    similar_items = self.menu_data[
                        (self.menu_data['Course'] == item_category) &
                        (self.menu_data['Name'] != item)
                    ]
                    if not similar_items.empty:
                        recommended_items.extend(similar_items.sample(min(2, len(similar_items)))['Name'].tolist())

    def show_recommendations_page(self, user_id):
        """Show full recommendations page"""
        # Clear current items
        for widget in self.menu_frame.winfo_children():
            widget.destroy()
        
        # Title
        CTkLabel(
            self.menu_frame,
            text="Personalized Recommendations",
            font=self.styles['heading']['font'],
            text_color="#000000"  # Black text
        ).pack(pady=20)
        
        # Get past orders based recommendations
        past_orders = self.backend.get_user_orders(user_id)
        recommended_items = []
        
        if past_orders:
            # Based on past orders
            past_items = []
            for _, items, _ in past_orders:
                past_items.extend(items.split(','))
            
            # Get similar items by category
            for item in past_items:
                item = item.strip()
                if item in self.menu_data['Name'].values:
                    item_category = self.menu_data[self.menu_data['Name'] == item]['Course'].iloc[0]
                    similar_items = self.menu_data[
                        (self.menu_data['Course'] == item_category) &
                        (self.menu_data['Name'] != item)
                    ]
                    if not similar_items.empty:
                        recommended_items.extend(similar_items.sample(min(3, len(similar_items)))['Name'].tolist())
        
        # Add general recommendations if needed
        if len(recommended_items) < 10:
            general_recs = self.backend.get_recommendations(user_id)
            recommended_items.extend(general_recs)
        
        # Remove duplicates and limit
        recommended_items = list(dict.fromkeys(recommended_items))[:10]
        
        if recommended_items:
            for item_name in recommended_items:
                item_data = self.menu_data[self.menu_data['Name'] == item_name].iloc[0]
                item_frame = CTkFrame(self.menu_frame, fg_color="#F5F5F5")
                item_frame.pack(fill="x", padx=10, pady=5)
                
                # Item details
                details_frame = CTkFrame(item_frame, fg_color="transparent")
                details_frame.pack(side="left", fill="x", expand=True, padx=10, pady=5)
                
                CTkLabel(
                    details_frame,
                    text=item_name,
                    font=self.styles['body']['font'],
                    text_color="#000000"  # Black text
                ).pack(anchor="w")
                
                CTkLabel(
                    details_frame,
                    text=f"Type: {item_data['Veg_Non']} | Course: {item_data['Course']}",
                    font=('Helvetica', 10),
                    text_color="#333333"  # Dark gray text
                ).pack(anchor="w")
                
                # Add to cart button
                CTkButton(
                    item_frame,
                    text="Add to Cart",
                    command=lambda i=item_data: self.add_to_cart(i, user_id),
                    fg_color="#4CAF50",
                    hover_color="#45a049",
                    width=100
                ).pack(side="right", padx=10)
        else:
            CTkLabel(
                self.menu_frame,
                text="No recommendations available yet.\nTry ordering some items!",
                font=self.styles['body']['font'],
                text_color="#333333"  # Dark gray text
            ).pack(pady=20)

if __name__ == "__main__":
    app = SmartFoodSystem()
    app.mainloop()