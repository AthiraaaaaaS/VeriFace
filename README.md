# VeriFace - an AI-Powered Attendance Monitoring System

VeriFace is an AI-Powered Attendance Monitoring Systemm that uses artificial intelligence and face recognition technology to automate the attendance tracking process. The system can detect and recognize multiple faces simultaneously, even at longer distances, making it ideal for various environments.

## Features

### Core Features

- **Multi-Face Detection**: Simultaneously detect and recognize multiple faces in a single frame
- **Long-Distance Recognition**: Improved face detection algorithms for recognizing faces at greater distances
- **Real-Time Attendance Tracking**: Automatically record attendance when faces are recognized
- **User Management**: Add, update, and delete users with their face data
- **Attendance Reports**: View and export attendance records to Excel
- **Confidence Scoring**: Each recognition includes a confidence score to ensure accuracy

### User Interface

- **Modern UI**: Clean, intuitive interface built with PyQt5
- **Multiple Screens**:
  - Login Screen: Secure access to the system
  - Attendance Screen: View and manage attendance records
  - Registration Screen: Add new users with face capture
  - User Management Screen: Manage existing users

### AI and Machine Learning

- **Face Recognition**: Uses InsightFace for accurate face detection and recognition
- **Machine Learning Models**:
  - SVM (Support Vector Machine) classifier for face recognition
  - KNN (K-Nearest Neighbors) classifier option available
- **Model Training**: Automatic model training when users are added or removed
- **Face Encoding**: Efficient face encoding extraction for database storage

### Database Management

- **SQLite Database**: Stores user information and attendance records
- **Data Integrity**: Maintains relationships between users and their attendance records
- **Automatic Cleanup**: Removes related attendance records when a user is deleted

### Export and Reporting

- **Excel Export**: Export attendance records to Excel for further analysis
- **Date Filtering**: Filter attendance records by date
- **Comprehensive Reports**: Includes user names, first seen, and last seen times

## Technical Details

### System Requirements

- Python 3.6+
- OpenCV
- PyQt5
- NumPy
- Pandas
- scikit-learn
- InsightFace
- SQLite3

### Directory Structure

- `models/`: Stores trained face recognition models
- `known_faces/`: Contains face images of registered users
- `face_data/`: Stores face encodings and labels
- `gui/`: Contains UI components and screens
- `attendance.db`: SQLite database for user and attendance data

### Key Components

- `face_recognition.py`: Core face recognition functionality
- `train_face_model.py`: Model training and management
- `attendance_ui.py`: User interface implementation
- `ai_integration.py`: AI system integration

## Usage

1. **Login**: Enter the admin password to access the system
2. **Register Users**: Add new users with their face images
3. **Record Attendance**: The system automatically recognizes faces and records attendance
4. **Manage Users**: Add, update, or delete users as needed
5. **View Reports**: Check attendance records and export to Excel

## Error Handling

The system includes comprehensive error handling for various scenarios:

- **Insufficient Users**: Clear messages when there aren't enough users to train the model
- **Face Detection Failures**: Helpful guidance when faces can't be detected in images
- **Database Errors**: Graceful handling of database connection and query issues
- **Model Training Failures**: Detailed error messages with traceback information

## Future Enhancements

- Mobile application for remote attendance tracking
- Cloud synchronization for multi-location deployment
- Advanced analytics and reporting features
- Integration with HR management systems
- Biometric authentication options
