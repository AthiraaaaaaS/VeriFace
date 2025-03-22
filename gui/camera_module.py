from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
from face_recognition_module import recognize_face
from gui.attendance_window import AttendanceWindow

class AttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance Monitoring System")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def setup_ui(self):
        """Sets up the UI Layout"""
        layout = QVBoxLayout()
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(640, 480)
        layout.addWidget(self.camera_label)

        # ✅ Admin Login
        self.password_field = QLineEdit()
        self.password_field.setPlaceholderText("Enter Admin Password")
        self.password_field.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_field)

        self.admin_login_button = QPushButton("Login as Admin")
        self.admin_login_button.clicked.connect(self.admin_login)
        layout.addWidget(self.admin_login_button)

        self.setLayout(layout)

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
