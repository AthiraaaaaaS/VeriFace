from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QLineEdit, QMessageBox, QDesktopWidget
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt
import cv2
from face_recognition_module import recognize_face
from gui.attendance_window import AttendanceWindow

class AttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Veriface")
        self.setFixedSize(400, 350)  # Reduced width and height
        self.center_on_screen()
        self.setup_ui()
        self.apply_styles()
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def center_on_screen(self):
        """Center the window on the screen"""
        frame_geometry = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

    def setup_ui(self):
        """Sets up the UI Layout"""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(10)  # Reduced spacing

        # Title
        title_label = QLabel("Welcome Back")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Enter your credentials to access your account")
        subtitle_label.setObjectName("subtitleLabel")
        layout.addWidget(subtitle_label)
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(640, 480)
        layout.addWidget(self.camera_label)

        # ✅ Admin Login
        self.password_field = QLineEdit()
        self.password_field.setPlaceholderText("Enter Admin Password")
        self.password_field.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_field)

        self.admin_login_button = QPushButton("Login as Admin")
        self.admin_login_button.setObjectName("signinButton")
        self.admin_login_button.clicked.connect(self.admin_login)
        layout.addWidget(self.admin_login_button)

        self.setLayout(layout)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                font-family: Arial, sans-serif;
            }
            
            #titleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2C3E50;
                margin-bottom: 5px;
                text-align: center;
            }

            #subtitleLabel {
                font-size: 14px;
                color: #7F8C8D;
                margin-bottom: 15px;
                text-align: center;
            }

            QLineEdit {
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                padding: 10px;
                font-size: 14px;
                color: #2C3E50;
                margin-bottom: 10px;
            }

            QLineEdit::placeholder {
                color: #BDC3C7;
            }

            #signinButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            #signinButton:hover {
                background-color: #2980B9;
            }
        """)
        
        # Add box shadow to the entire window
        self.setStyleSheet(self.styleSheet() + """
            QWidget {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            }
        """)

    def start_camera(self):
        """Opens the webcam."""
        self.capture = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        """Updates the camera feed."""
        ret, frame = self.capture.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_qt)
            self.camera_label.setPixmap(pixmap)

    def admin_login(self):
        """Opens attendance log if the admin password is correct."""
        if self.password_field.text() == "admin123":
            self.attendance_window = AttendanceWindow()
            self.attendance_window.show()
        else:
            QMessageBox.warning(self, "Access Denied", "Incorrect Password")

if __name__ == "__main__":
    app = QApplication([])
    window = AttendanceApp()
    window.show()
    app.exec_()
