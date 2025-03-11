import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from face_recognition_module import recognize_face

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
        layout = QVBoxLayout()
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(640, 480)
        layout.addWidget(self.camera_label)
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        layout.addWidget(self.start_button)
        self.capture_button = QPushButton("Mark Attendance")
        self.capture_button.clicked.connect(self.capture_frame)
        layout.addWidget(self.capture_button)
        self.retake_button = QPushButton("Retake Image")
        self.retake_button.clicked.connect(self.start_camera)
        layout.addWidget(self.retake_button)
        self.setLayout(layout)

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_qt)
            self.camera_label.setPixmap(pixmap)

    def capture_frame(self):
        ret, frame = self.capture.read()
        if ret:
            name = recognize_face(frame)
            if name and name != "Unknown":
                QMessageBox.information(self, "Attendance Marked", f"Attendance recorded for {name}")
            else:
                QMessageBox.warning(self, "No Match", "Face not recognized. Please try again.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceApp()
    window.show()
    sys.exit(app.exec_())