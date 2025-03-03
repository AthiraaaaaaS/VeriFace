VeriFace: Attendance Monitoring System Using Face ID

📌 Introduction

VeriFace is an AI-powered attendance monitoring system that uses face recognition technology to automate attendance marking. It enhances accuracy, eliminates manual effort, and prevents proxy attendance.

The system is built using:

dlib’s ResNet-34 deep metric learning model for face recognition.

OpenCV for real-time video capture.

Python for backend logic.

CSV (with future SQLite integration) for attendance storage.

🚀 Features

✅ Real-time face detection and recognition✅ Automated attendance marking✅ Uses Euclidean distance for face matching✅ Prevents duplicate entries for the same day✅ Supports database integration (future scope)

🛠️ System Architecture

1️⃣ User Interface (GUI & Camera Module): Captures live video2️⃣ Face Recognition Module: Detects and recognizes faces3️⃣ Attendance Marking Module: Logs attendance in CSV4️⃣ Utility Module: Handles errors and file management5️⃣ Database (CSV / Future SQLite): Stores attendance records

🏗️ Installation Guide

🔹 Prerequisites

Ensure you have the following installed:

Python 3.x

OpenCV (pip install opencv-python)

face_recognition (pip install face-recognition)

dlib (pip install dlib)

NumPy (pip install numpy)

🔹 Clone the Repository

git clone https://github.com/yourusername/VeriFace.git
cd VeriFace

🔹 Run the Application

python main.py

🎯 How It Works

1️⃣ Run the program using python main.py.2️⃣ Webcam captures a live video frame.3️⃣ Face detection is performed using OpenCV & dlib.4️⃣ Feature extraction (128D vector) is done using ResNet-34.5️⃣ Face matching is performed using Euclidean distance.6️⃣ If a match is found, attendance is marked in attendance.csv.

📂 Project Structure

VeriFace/
│── known_faces/           # Folder to store known face images
│── attendance.csv         # CSV file to log attendance
│── main.py                # Entry point of the system
│── attendance.py          # Attendance marking module
│── face_recognition_model.py  # Face recognition functions
│── utils.py               # Utility functions
│── README.md              # Project documentation

🛠️ Future Enhancements

✅ Upgrade CSV storage to SQLite/MySQL✅ Add a PyQt-based GUI✅ Develop a mobile app for attendance tracking✅ Improve accuracy with advanced AI models

🤝 Contributing

Feel free to contribute by creating pull requests. For major changes, open an issue first to discuss your ideas.

📜 License

This project is licensed under the MIT License.

📞 Contact

For any queries, reach out to your.email@example.com or visit https://github.com/yourusername/VeriFace.

🚀 Start using VeriFace today for seamless, AI-powered attendance tracking!

