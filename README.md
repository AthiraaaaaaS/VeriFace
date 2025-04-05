# VeriFace : An AI powered Attendance Monitoring System

## Overview
The Attendance Monitoring System is a real-time face recognition application built using **PyQt5**, **OpenCV**, **InsightFace**, **ArcFace ONNX**, and **SQLite**. It allows organizations to monitor attendance using **automated facial recognition**. The system features **live camera monitoring**, an **admin panel**, and an **attendance tracking database**.

---

## Features

- **Real-Time Face Recognition:** Uses **InsightFace** and **ArcFace ONNX** models for high-accuracy face detection and recognition.
- **Live Camera Monitoring:** Continuous video feed to detect and recognize faces.
- **Attendance Tracking:** Logs **first seen** and **last seen** timestamps for each user.
- **Admin Panel:** Secure login for viewing attendance records.
- **SQLite Database:** Stores user face encodings and attendance logs efficiently.

---

## Technologies Used

- **Python 3.x**
- **PyQt5** – GUI framework
- **OpenCV** – Computer vision and face detection
- **InsightFace** – High-accuracy face recognition
- **ArcFace ONNX** – Face recognition model
- **SQLite3** – Database for storing user information and attendance logs

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/VeriFace.git
cd VeriFace

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/VeriFace.git
cd VeriFace
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
VeriFace/
├── models/                     # Pre-trained face recognition models
│   ├── arcface.onnx            #download link https://huggingface.co/FoivosPar/Arc2Face/blob/da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx
│   ├── opencv_face_detector.pb
|   ├── opencv_face_detector.pbtxt
├── gui/                        # GUI components
│   ├── attendance_ui.py  
├── known_faces/                # Folder for storing user images
├── attendance.db               # SQLite database
├── attendance.py               # Handles attendance logging
├── face_recognition_module.py  # Recognizes faces from live video feed
├── live_camera.py              # Background CCTV-style live monitoring
├── main.py                     # Launches the application
├── utils.py                    # Utility functions for database operations
├── add_user.py                 # Adds new users to the database
├── requirements.txt            # Required Python dependencies
└── README.md                   # Project documentation
```

---

## Usage

### 1. Adding a User
To add a new user with their face encoding:
```bash
python add_user.py "John Doe" "known_faces/johndoe.jpg"
```

### 2. Running the Application
```bash
python main.py
```
This will:
- Open the **camera interface**
- Recognize faces and **log attendance**
- Allow admins to **view attendance records**

### 3. Admin Login
- Click **"Login as Admin"**
- Enter the password (**default: admin123**)
- View **first seen** and **last seen** timestamps

---

## Database Schema

### Users Table (`users`)
| Column  | Type    | Description        |
|---------|--------|--------------------|
| `id`    | INTEGER | Unique user ID |
| `name`  | TEXT   | User’s full name  |
| `encoding` | BLOB | Face encoding for recognition |
| `email`  | TEXT   | User’s email id  |
| `phone` | TEXT | User’s phone number |


### Attendance Table (`attendance`)
| Column  | Type    | Description        |
|---------|--------|--------------------|
| `id`    | INTEGER | Unique record ID |
| `user_id` | INTEGER | References `users.id` |
| `first_seen` | TEXT | First time user was detected |
| `last_seen` | TEXT | Last recorded presence |

---

## Future Enhancements

- **Multi-camera support** for large-scale implementations.
- **Cloud integration** for remote attendance tracking.
- **Mobile application** for on-the-go access.
- **Integration with RFID and biometrics** for multi-factor authentication.

---

## License
This updated README reflects the use of **InsightFace** and **ArcFace** for face recognition, as well as the other recent changes. Let me know if you need further edits!


