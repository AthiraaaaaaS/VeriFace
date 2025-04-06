from PyQt5.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QDateEdit, QFileDialog, QMessageBox, QHeaderView, QHBoxLayout, QFormLayout, QFrame
from PyQt5.QtCore import Qt, QDate, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage, QColor, QPalette
import sqlite3
import pandas as pd
import os
import cv2
from datetime import datetime
import numpy as np
import sys

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition import FaceRecognizer
from train_face_model import train_and_save_face_model

# Global styles
GLOBAL_STYLE = """
    QWidget {
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    QPushButton {
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 14px;
    }
    
    QLineEdit {
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 8px 12px;
        background-color: #FFFFFF;
        selection-background-color: #2196F3;
    }
    
    QLineEdit:focus {
        border: 1px solid #2196F3;
    }
    
    QTableWidget {
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        background-color: #FFFFFF;
        gridline-color: #F5F5F5;
    }
    
    QHeaderView::section {
        background-color: #F5F5F5;
        color: #424242;
        padding: 10px;
        border: none;
        font-weight: 600;
        font-size: 13px;
    }
    
    QTableWidget::item {
        padding: 8px;
        border-bottom: 1px solid #F5F5F5;
    }
    
    QTableWidget::item:selected {
        background-color: #E3F2FD;
        color: #1976D2;
    }
    
    QDateEdit {
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 8px 12px;
        background-color: #FFFFFF;
    }
    
    QDateEdit::drop-down {
        border: none;
        width: 20px;
    }
    
    QDateEdit::down-arrow {
        image: url(icons/calendar.png);
        width: 12px;
        height: 12px;
    }
"""

class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VeriFace - AI Powered Attendance Monitoring System")
        self.setGeometry(100, 100, 1000, 700)  # Increased window size
        
        # Set application style
        self.setStyleSheet(GLOBAL_STYLE)
        
        # Load and display logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons","veriface.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        else:
            print("Logo file not found.")
        
        self.login_screen = LoginScreen(self)
        self.attendance_screen = AttendanceScreen(self)
        self.register_screen = RegisterScreen(self)
        self.users_screen = UsersScreen(self)
        
        self.addWidget(self.login_screen)
        self.addWidget(self.attendance_screen)
        self.addWidget(self.register_screen)
        self.addWidget(self.users_screen)
        
        self.setCurrentWidget(self.login_screen)

class LoginScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()
        
    def setup_ui(self):
        # Set background color
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F7FA;
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        main_layout.setAlignment(Qt.AlignCenter)
        
        # Card container
        card = QFrame()
        card.setObjectName("loginCard")
        card.setStyleSheet("""
            #loginCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)
        
        # Load and display logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons","veriface.png")
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_label = QLabel()
            logo_label.setPixmap(logo_pixmap.scaled(150, 150, Qt.KeepAspectRatio))
            logo_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(logo_label)
        else:
            print("Logo file not found.")

        # Title
        title_label = QLabel("Welcome to VeriFace")
        title_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #1976D2;")
        card_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("AI Powered Attendance Monitoring System")
        subtitle_label.setFont(QFont("Segoe UI", 14))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #757575;")
        card_layout.addWidget(subtitle_label)
        
        # Spacer
        card_layout.addSpacing(20)
        
        # Password field
        self.password_field = QLineEdit()
        self.password_field.setPlaceholderText("Enter Admin Password")
        self.password_field.setEchoMode(QLineEdit.Password)
        self.password_field.setMinimumHeight(45)
        self.password_field.setStyleSheet("""
            QLineEdit {
                font-size: 16px;
            }
        """)
        card_layout.addWidget(self.password_field)
        
        # Login button
        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.admin_login)
        self.login_button.setObjectName("loginButton")
        self.login_button.setMinimumHeight(45)
        self.login_button.setStyleSheet("""
            #loginButton {
                background-color: #1976D2;
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
            
            #loginButton:hover {
                background-color: #1565C0;
            }
        """)
        card_layout.addWidget(self.login_button)
        
        # Add card to main layout
        main_layout.addWidget(card)
        
        self.setLayout(main_layout)
        
    def admin_login(self):
        if self.password_field.text() == "admin123":
            self.main_window.setCurrentWidget(self.main_window.attendance_screen)
        else:
            QMessageBox.warning(self, "Access Denied", "Incorrect Password")

class AttendanceScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()
        
    def setup_ui(self):
        # Set background color
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F7FA;
            }
        """)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Top card with title and buttons
        top_card = QFrame()
        top_card.setObjectName("topCard")
        top_card.setStyleSheet("""
            #topCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        
        top_layout = QVBoxLayout(top_card)
        top_layout.setContentsMargins(20, 20, 20, 20)
        top_layout.setSpacing(15)
        
        # Title and buttons row
        title_buttons_layout = QHBoxLayout()
        
        # Left side - Title
        self.title_label = QLabel("Attendance Records")
        self.title_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #1976D2;")
        title_buttons_layout.addWidget(self.title_label)
        
        # Center - Buttons
        buttons_layout = QHBoxLayout()
        
        # Register button
        self.register_button = QPushButton("New User/Register")
        self.register_button.clicked.connect(self.show_register_screen)
        self.register_button.setObjectName("registerButton")
        self.register_button.setStyleSheet("""
            #registerButton {
                background-color: #4CAF50;
                color: white;
            }
            
            #registerButton:hover {
                background-color: #43A047;
            }
        """)
        buttons_layout.addWidget(self.register_button)
        
        # Users button
        self.users_button = QPushButton("Manage Users")
        self.users_button.clicked.connect(self.show_users_screen)
        self.users_button.setObjectName("usersButton")
        self.users_button.setStyleSheet("""
            #usersButton {
                background-color: #FF9800;
                color: white;
            }
            
            #usersButton:hover {
                background-color: #F57C00;
            }
        """)
        buttons_layout.addWidget(self.users_button)
        
        title_buttons_layout.addLayout(buttons_layout)
        
        # Right side - Logout button
        self.logout_button = QPushButton("Logout")
        self.logout_button.clicked.connect(self.logout)
        self.logout_button.setObjectName("logoutButton")
        self.logout_button.setStyleSheet("""
            #logoutButton {
                background-color: #F44336;
                color: white;
            }
            
            #logoutButton:hover {
                background-color: #E53935;
            }
        """)
        title_buttons_layout.addWidget(self.logout_button)
        
        top_layout.addLayout(title_buttons_layout)
        
        # Date filter row
        date_filter_layout = QHBoxLayout()
        
        date_label = QLabel("Select Date:")
        date_label.setStyleSheet("font-weight: bold; color: #424242;")
        date_filter_layout.addWidget(date_label)
        
        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())
        self.date_picker.setObjectName("datePicker")
        self.date_picker.setStyleSheet("""
            QDateEdit {
                min-width: 200px;
            }
        """)
        date_filter_layout.addWidget(self.date_picker)
        
        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self.display_attendance_table)
        self.filter_button.setObjectName("filterButton")
        self.filter_button.setStyleSheet("""
            #filterButton {
                background-color: #2196F3;
                color: white;
            }
            
            #filterButton:hover {
                background-color: #1E88E5;
            }
        """)
        date_filter_layout.addWidget(self.filter_button)
        date_filter_layout.addStretch()
        
        top_layout.addLayout(date_filter_layout)
        
        layout.addWidget(top_card)
        
        # Attendance table card
        table_card = QFrame()
        table_card.setObjectName("tableCard")
        table_card.setStyleSheet("""
            #tableCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(20, 20, 20, 20)
        table_layout.setSpacing(15)
        
        # Table title
        table_title = QLabel("Today's Attendance")
        table_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        table_title.setStyleSheet("color: #424242;")
        table_layout.addWidget(table_title)
        
        # Attendance table
        self.attendance_table = QTableWidget()
        self.attendance_table.setColumnCount(3)
        self.attendance_table.setHorizontalHeaderLabels(["Name", "First Seen", "Last Seen"])
        self.attendance_table.setStyleSheet("""
            QTableWidget {
                border: none;
                gridline-color: #F5F5F5;
            }
            
            QHeaderView::section {
                background-color: #F5F5F5;
                color: #424242;
                padding: 12px;
                border: none;
                font-weight: 600;
                font-size: 13px;
            }
            
            QTableWidget::item {
                padding: 12px;
                border-bottom: 1px solid #F5F5F5;
            }
            
            QTableWidget::item:selected {
                background-color: #E3F2FD;
                color: #1976D2;
            }
        """)
        header = self.attendance_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.attendance_table)
        
        # Export button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_button = QPushButton("Export to Excel")
        self.export_button.clicked.connect(self.export_to_excel)
        self.export_button.setObjectName("exportButton")
        self.export_button.setStyleSheet("""
            #exportButton {
                background-color: #4CAF50;
                color: white;
            }
            
            #exportButton:hover {
                background-color: #43A047;
            }
        """)
        export_layout.addWidget(self.export_button)
        table_layout.addLayout(export_layout)
        
        layout.addWidget(table_card)
        
        self.setLayout(layout)
        
        # Display attendance on initialization
        self.display_attendance_table()
        
    def display_attendance_table(self):
        selected_date = self.date_picker.date().toString("yyyy-MM-dd")
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        query = """
        SELECT users.name, attendance.first_seen, attendance.last_seen 
        FROM attendance
        INNER JOIN users ON attendance.user_id = users.id
        WHERE DATE(attendance.first_seen) = ?
        """
        cursor.execute(query, (selected_date,))
        records = cursor.fetchall()
        conn.close()
        
        self.attendance_table.setRowCount(len(records))
        for row_idx, (name, first_seen, last_seen) in enumerate(records):
            self.attendance_table.setItem(row_idx, 0, QTableWidgetItem(name))
            self.attendance_table.setItem(row_idx, 1, QTableWidgetItem(first_seen))
            self.attendance_table.setItem(row_idx, 2, QTableWidgetItem(last_seen))
        
    def export_to_excel(self):
        row_count = self.attendance_table.rowCount()

        if row_count == 0:
            QMessageBox.warning(self, "No Data", "No attendance records available to export.")
            return

        data = []
        for row in range(row_count):
            row_data = []
            for col in range(self.attendance_table.columnCount()):
                item = self.attendance_table.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append("")  # Handle empty cells
            data.append(row_data)

        df = pd.DataFrame(data, columns=["Name", "First Seen", "Last Seen"])
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Attendance", "", "Excel Files (*.xlsx);;All Files (*)")
        if file_path:
            if not file_path.endswith(".xlsx"):
                file_path += ".xlsx"
            df.to_excel(file_path, index=False, engine="openpyxl")
            QMessageBox.information(self, "Export Successful", f"Attendance saved to:\n{file_path}")
        
    def logout(self):
        self.main_window.setCurrentWidget(self.main_window.login_screen)
        self.main_window.login_screen.password_field.clear()
    
    def show_register_screen(self):
        self.main_window.setCurrentWidget(self.main_window.register_screen)

    def show_users_screen(self):
        self.main_window.setCurrentWidget(self.main_window.users_screen)

class RegisterScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.captured_image = None
        self.setup_ui()
        
    def setup_ui(self):
        # Set background color
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F7FA;
            }
        """)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Top card with title and back button
        top_card = QFrame()
        top_card.setObjectName("topCard")
        top_card.setStyleSheet("""
            #topCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        
        top_layout = QHBoxLayout(top_card)
        top_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Register New User")
        title_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title_label.setStyleSheet("color: #1976D2;")
        top_layout.addWidget(title_label)
        
        # Back button
        self.back_button = QPushButton("Back to Attendance")
        self.back_button.clicked.connect(self.go_back)
        self.back_button.setObjectName("backButton")
        self.back_button.setStyleSheet("""
            #backButton {
                background-color: #757575;
                color: white;
            }
            
            #backButton:hover {
                background-color: #616161;
            }
        """)
        top_layout.addWidget(self.back_button)
        
        layout.addWidget(top_card)
        
        # Form card
        form_card = QFrame()
        form_card.setObjectName("formCard")
        form_card.setStyleSheet("""
            #formCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(30, 30, 30, 30)
        form_layout.setSpacing(20)
        
        # Form title
        form_title = QLabel("User Information")
        form_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        form_title.setStyleSheet("color: #424242;")
        form_layout.addWidget(form_title)
        
        # Form fields
        fields_layout = QFormLayout()
        fields_layout.setSpacing(15)
        fields_layout.setLabelAlignment(Qt.AlignRight)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter full name")
        self.name_input.setMinimumHeight(40)
        
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Enter email address")
        self.email_input.setMinimumHeight(40)
        
        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("Enter phone number")
        self.phone_input.setMinimumHeight(40)
        
        fields_layout.addRow("Name:", self.name_input)
        fields_layout.addRow("Email:", self.email_input)
        fields_layout.addRow("Phone:", self.phone_input)
        
        form_layout.addLayout(fields_layout)
        
        # Image capture section
        image_section = QFrame()
        image_section.setObjectName("imageSection")
        image_section.setStyleSheet("""
            #imageSection {
                background-color: #F5F7FA;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        
        image_layout = QVBoxLayout(image_section)
        image_layout.setContentsMargins(20, 20, 20, 20)
        image_layout.setSpacing(15)
        
        image_title = QLabel("Face Image")
        image_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        image_title.setStyleSheet("color: #424242;")
        image_layout.addWidget(image_title)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(200, 200)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
            }
        """)
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        
        # Capture button
        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setObjectName("captureButton")
        self.capture_button.setStyleSheet("""
            #captureButton {
                background-color: #2196F3;
                color: white;
            }
            
            #captureButton:hover {
                background-color: #1E88E5;
            }
        """)
        image_layout.addWidget(self.capture_button, alignment=Qt.AlignCenter)
        
        form_layout.addWidget(image_section)
        
        # Register button
        self.register_button = QPushButton("Register User")
        self.register_button.clicked.connect(self.register_user)
        self.register_button.setObjectName("registerButton")
        self.register_button.setMinimumHeight(45)
        self.register_button.setStyleSheet("""
            #registerButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
            
            #registerButton:hover {
                background-color: #43A047;
            }
        """)
        form_layout.addWidget(self.register_button)
        
        layout.addWidget(form_card)
        
        self.setLayout(layout)
        
    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Could not open camera")
            return
            
        ret, frame = cap.read()
        if ret:
            self.captured_image = frame
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            
        cap.release()
        
    def register_user(self):
        if self.captured_image is None:
            QMessageBox.warning(self, "Error", "Please capture an image first")
            return
            
        name = self.name_input.text()
        email = self.email_input.text()
        phone = self.phone_input.text()
        
        if not all([name, email, phone]):
            QMessageBox.warning(self, "Error", "Please fill in all fields")
            return
            
        try:
            # Save image in known_faces folder
            image_dir = "known_faces"
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
                
            # Clean the name to create a valid filename
            clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            image_path = os.path.join(image_dir, f"{clean_name}.jpg")
            
            # Check if file already exists
            if os.path.exists(image_path):
                reply = QMessageBox.question(self, 'File Exists', 
                    f'An image for {name} already exists. Do you want to replace it?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
                if reply == QMessageBox.No:
                    return
            
            cv2.imwrite(image_path, self.captured_image)
            
            # Get face encoding using FaceRecognizer
            recognizer = FaceRecognizer()
            face_encoding = recognizer.extract_face_encoding(image_path)
            if face_encoding is None:
                QMessageBox.warning(self, "Error", "Could not detect face in the image. Please try again.")
                return
                
            face_encoding_blob = face_encoding.astype(np.float32).tobytes()
            
            # Save to database
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            
            # Create users table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    encoding BLOB,
                    phone TEXT,
                    email TEXT
                )
            """)
            
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                # Update existing user
                user_id = existing_user[0]
                cursor.execute("""
                    UPDATE users 
                    SET encoding = ?, phone = ?, email = ?
                    WHERE name = ?
                """, (face_encoding_blob, phone, email, name))
            else:
                # Insert new user
                cursor.execute("""
                    INSERT INTO users (name, encoding, phone, email)
                    VALUES (?, ?, ?, ?)
                """, (name, face_encoding_blob, phone, email))
                user_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            QMessageBox.information(self, "Success", f"User {name} registered successfully!")
            
            # Train the model with the new data
            self.train_model()
            
            # Clear fields
            self.clear_fields()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            
    def train_model(self):
        """Train the face recognition model after registering a user."""
        try:
            # Show a progress message
            progress_msg = QMessageBox()
            progress_msg.setIcon(QMessageBox.Information)
            progress_msg.setText("Training AI Model")
            progress_msg.setInformativeText("Please wait while the face recognition model is being trained...")
            progress_msg.setWindowTitle("Processing")
            progress_msg.show()
            
            # Train the model
            result = train_and_save_face_model(classifier_type='svm')
            
            # Close the progress message
            progress_msg.close()
            
            if result['success']:
                QMessageBox.information(self, "Success", "User registered and model trained successfully!")
            else:
                QMessageBox.warning(self, "Warning", 
                    f"User registered, but model training failed: {result['message']}")
            
            # Clear fields and go back
            self.clear_fields()
            self.go_back()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"Failed to train AI model: {str(e)}\n\nPlease check the console for more details.")
            print(f"Error details: {error_details}")
            self.clear_fields()
            self.go_back()
            
    def clear_fields(self):
        self.name_input.clear()
        self.email_input.clear()
        self.phone_input.clear()
        self.image_label.clear()
        self.captured_image = None
        
    def go_back(self):
        self.clear_fields()
        self.main_window.setCurrentWidget(self.main_window.attendance_screen)

class UsersScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()
        
    def setup_ui(self):
        # Set background color
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F7FA;
            }
        """)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Top card with title and back button
        top_card = QFrame()
        top_card.setObjectName("topCard")
        top_card.setStyleSheet("""
            #topCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        
        top_layout = QHBoxLayout(top_card)
        top_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        self.title_label = QLabel("User Management")
        self.title_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #1976D2;")
        top_layout.addWidget(self.title_label)
        
        # Back button
        self.back_button = QPushButton("Back to Attendance")
        self.back_button.clicked.connect(self.go_back)
        self.back_button.setObjectName("backButton")
        self.back_button.setStyleSheet("""
            #backButton {
                background-color: #757575;
                color: white;
            }
            
            #backButton:hover {
                background-color: #616161;
            }
        """)
        top_layout.addWidget(self.back_button)
        
        layout.addWidget(top_card)
        
        # Users table card
        table_card = QFrame()
        table_card.setObjectName("tableCard")
        table_card.setStyleSheet("""
            #tableCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(20, 20, 20, 20)
        table_layout.setSpacing(15)
        
        # Table title
        table_title = QLabel("Registered Users")
        table_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        table_title.setStyleSheet("color: #424242;")
        table_layout.addWidget(table_title)
        
        # Users table
        self.users_table = QTableWidget()
        self.users_table.setColumnCount(5)
        self.users_table.setHorizontalHeaderLabels(["ID", "Name", "Email", "Phone", "Actions"])
        self.users_table.setStyleSheet("""
            QTableWidget {
                border: none;
                gridline-color: #F5F5F5;
            }
            
            QHeaderView::section {
                background-color: #F5F5F5;
                color: #424242;
                padding: 12px;
                border: none;
                font-weight: 600;
                font-size: 13px;
            }
            
            QTableWidget::item {
                padding: 12px;
                border-bottom: 1px solid #F5F5F5;
            }
            
            QTableWidget::item:selected {
                background-color: #E3F2FD;
                color: #1976D2;
            }
        """)
        header = self.users_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.users_table)
        
        # Refresh button
        refresh_layout = QHBoxLayout()
        refresh_layout.addStretch()
        self.refresh_button = QPushButton("Refresh Users")
        self.refresh_button.clicked.connect(self.load_users)
        self.refresh_button.setObjectName("refreshButton")
        self.refresh_button.setStyleSheet("""
            #refreshButton {
                background-color: #2196F3;
                color: white;
            }
            
            #refreshButton:hover {
                background-color: #1E88E5;
            }
        """)
        refresh_layout.addWidget(self.refresh_button)
        table_layout.addLayout(refresh_layout)
        
        layout.addWidget(table_card)
        
        self.setLayout(layout)
        
        # Load users on initialization
        self.load_users()
        
    def load_users(self):
        """Load all users from the database and display them in the table."""
        try:
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            
            # Get all users
            cursor.execute("SELECT id, name, email, phone FROM users")
            users = cursor.fetchall()
            
            conn.close()
            
            # Clear the table
            self.users_table.setRowCount(0)
            
            # Add users to the table
            for row_idx, (user_id, name, email, phone) in enumerate(users):
                self.users_table.insertRow(row_idx)
                
                # Add user data
                self.users_table.setItem(row_idx, 0, QTableWidgetItem(str(user_id)))
                self.users_table.setItem(row_idx, 1, QTableWidgetItem(name))
                self.users_table.setItem(row_idx, 2, QTableWidgetItem(email))
                self.users_table.setItem(row_idx, 3, QTableWidgetItem(phone))
                
                # Add delete button
                delete_button = QPushButton("Delete")
                delete_button.setObjectName("deleteButton")
                delete_button.clicked.connect(lambda checked, uid=user_id: self.delete_user(uid))
                delete_button.setStyleSheet("""
                    #deleteButton {
                        background-color: #F44336;
                        color: white;
                        padding: 5px 10px;
                        font-size: 12px;
                    }
                    
                    #deleteButton:hover {
                        background-color: #E53935;
                    }
                """)
                
                # Create a widget to hold the button
                button_widget = QWidget()
                button_layout = QHBoxLayout(button_widget)
                button_layout.addWidget(delete_button)
                button_layout.setAlignment(Qt.AlignCenter)
                button_layout.setContentsMargins(0, 0, 0, 0)
                
                self.users_table.setCellWidget(row_idx, 4, button_widget)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load users: {str(e)}")
    
    def delete_user(self, user_id):
        """Delete a user from the database."""
        try:
            # Confirm deletion
            reply = QMessageBox.question(
                self, 
                "Confirm Deletion", 
                "Are you sure you want to delete this user? This action cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                conn = sqlite3.connect("attendance.db")
                cursor = conn.cursor()
                
                # Get user name for confirmation message
                cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
                user = cursor.fetchone()
                user_name = user[0] if user else "Unknown"
                
                # Delete user
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                
                # Delete attendance records
                cursor.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
                
                conn.commit()
                conn.close()
                
                QMessageBox.information(self, "Success", f"User {user_name} has been deleted.")
                
                # Reload users
                self.load_users()
                
                # Retrain the model
                self.train_model()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete user: {str(e)}")
    
    def train_model(self):
        """Train the face recognition model after deleting a user."""
        try:
            # Check if there are enough users to train the model
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            conn.close()
            
            if user_count < 2:
                QMessageBox.information(self, "Information", 
                    "Not enough users to train the model. At least 2 users are required.\n\nPlease add more users before training the model.")
                return
            
            # Show a progress message
            progress_msg = QMessageBox()
            progress_msg.setIcon(QMessageBox.Information)
            progress_msg.setText("Training AI Model")
            progress_msg.setInformativeText("Please wait while the face recognition model is being trained...")
            progress_msg.setWindowTitle("Processing")
            progress_msg.show()
            
            # Train the model
            result = train_and_save_face_model(classifier_type='svm')
            
            # Close the progress message
            progress_msg.close()
            
            if result['success']:
                QMessageBox.information(self, "Success", "Model trained successfully!")
            else:
                QMessageBox.warning(self, "Warning", 
                    f"Model training failed: {result['message']}")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"Failed to train AI model: {str(e)}\n\nPlease check the console for more details.")
            print(f"Error details: {error_details}")
    
    def go_back(self):
        self.main_window.setCurrentWidget(self.main_window.attendance_screen)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()