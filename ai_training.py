import os
import numpy as np
import cv2
import face_recognition
import pickle
from datetime import datetime
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VeriFace_AI')

class VeriFaceAI:
    def __init__(self, model_path='models/face_recognition_model.pkl', 
                 known_faces_dir='known_faces', 
                 db_path='attendance.db'):
        """
        Initialize the VeriFace AI training system
        
        Args:
            model_path: Path to save/load the trained model
            known_faces_dir: Directory containing known face images
            db_path: Path to the SQLite database
        """
        self.model_path = model_path
        self.known_faces_dir = known_faces_dir
        self.db_path = db_path
        self.model = None
        self.face_encodings = []
        self.face_labels = []
        self.attendance_data = []
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Load existing model if available
        self.load_model()
        
    def load_model(self):
        """Load the trained model if it exists"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            logger.info("No existing model found. Creating new model.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def save_model(self):
        """Save the trained model to disk"""
        try:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def extract_face_encoding(self, image_path):
        """
        Extract face encoding from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Face encoding as numpy array or None if no face detected
        """
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                logger.warning(f"No face detected in {image_path}")
                return None
                
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                logger.warning(f"Could not encode face in {image_path}")
                return None
                
            # Return the first face encoding
            return face_encodings[0]
            
        except Exception as e:
            logger.error(f"Error extracting face encoding: {e}")
            return None
    
    def collect_training_data(self):
        """
        Collect training data from known faces directory and database
        """
        logger.info("Collecting training data...")
        
        # Reset data
        self.face_encodings = []
        self.face_labels = []
        
        # Collect face encodings from known faces directory
        if os.path.exists(self.known_faces_dir):
            for filename in os.listdir(self.known_faces_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    # Extract user ID from filename (assuming format: user_id.jpg)
                    user_id = os.path.splitext(filename)[0]
                    
                    # Get face encoding
                    image_path = os.path.join(self.known_faces_dir, filename)
                    face_encoding = self.extract_face_encoding(image_path)
                    
                    if face_encoding is not None:
                        self.face_encodings.append(face_encoding)
                        self.face_labels.append(user_id)
                        logger.info(f"Added training data for user {user_id}")
        
        # Collect attendance data from database
        self.collect_attendance_data()
        
        logger.info(f"Collected {len(self.face_encodings)} face encodings for training")
    
    def collect_attendance_data(self):
        """
        Collect attendance data from the database for pattern analysis
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get attendance records with user information
            cursor.execute("""
                SELECT users.id, users.name, attendance.first_seen, attendance.last_seen
                FROM attendance
                JOIN users ON attendance.user_id = users.id
                ORDER BY attendance.first_seen
            """)
            
            records = cursor.fetchall()
            
            # Process records into structured data
            self.attendance_data = []
            for user_id, name, first_seen, last_seen in records:
                # Convert string timestamps to datetime objects
                first_seen_dt = datetime.strptime(first_seen, "%Y-%m-%d %H:%M:%S")
                last_seen_dt = datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S")
                
                # Calculate duration in minutes
                duration = (last_seen_dt - first_seen_dt).total_seconds() / 60
                
                # Extract day of week (0 = Monday, 6 = Sunday)
                day_of_week = first_seen_dt.weekday()
                
                # Extract hour of day (0-23)
                hour_of_day = first_seen_dt.hour
                
                self.attendance_data.append({
                    'user_id': user_id,
                    'name': name,
                    'date': first_seen_dt.date(),
                    'day_of_week': day_of_week,
                    'hour_of_day': hour_of_day,
                    'duration': duration
                })
            
            conn.close()
            logger.info(f"Collected {len(self.attendance_data)} attendance records")
            
        except Exception as e:
            logger.error(f"Error collecting attendance data: {e}")
    
    def train_face_recognition_model(self):
        """
        Train the face recognition model using collected data
        """
        if not self.face_encodings:
            logger.warning("No training data available. Please collect data first.")
            return False
            
        try:
            # Convert face encodings to numpy array
            X = np.array(self.face_encodings)
            y = np.array(self.face_labels)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            logger.info(f"Model trained successfully. Train score: {train_score:.4f}, Test score: {test_score:.4f}")
            
            # Save the trained model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def train_attendance_prediction_model(self):
        """
        Train a model to predict attendance patterns
        """
        if not self.attendance_data:
            logger.warning("No attendance data available. Please collect data first.")
            return False
            
        try:
            # Group data by user
            user_data = {}
            for record in self.attendance_data:
                user_id = record['user_id']
                if user_id not in user_data:
                    user_data[user_id] = []
                user_data[user_id].append(record)
            
            # Train a model for each user
            user_models = {}
            for user_id, records in user_data.items():
                if len(records) < 5:  # Skip users with too few records
                    continue
                    
                # Extract features
                X = []
                y = []
                
                for record in records:
                    # Features: day of week, hour of day
                    X.append([record['day_of_week'], record['hour_of_day']])
                    # Target: duration
                    y.append(record['duration'])
                
                # Train model
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)
                
                user_models[user_id] = model
                logger.info(f"Trained attendance prediction model for user {user_id}")
            
            # Save user models
            os.makedirs('models/attendance', exist_ok=True)
            for user_id, model in user_models.items():
                model_path = f'models/attendance/user_{user_id}_model.pkl'
                joblib.dump(model, model_path)
            
            logger.info(f"Trained attendance prediction models for {len(user_models)} users")
            return True
            
        except Exception as e:
            logger.error(f"Error training attendance prediction model: {e}")
            return False
    
    def predict_attendance(self, user_id, day_of_week, hour_of_day):
        """
        Predict attendance duration for a user
        
        Args:
            user_id: User ID
            day_of_week: Day of week (0-6, where 0 is Monday)
            hour_of_day: Hour of day (0-23)
            
        Returns:
            Predicted duration in minutes or None if prediction failed
        """
        try:
            model_path = f'models/attendance/user_{user_id}_model.pkl'
            
            if not os.path.exists(model_path):
                logger.warning(f"No prediction model found for user {user_id}")
                return None
                
            model = joblib.load(model_path)
            prediction = model.predict([[day_of_week, hour_of_day]])[0]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting attendance: {e}")
            return None
    
    def update_model_with_new_data(self, face_image_path, user_id, silent=False):
        """
        Update the model with new face data
        
        Args:
            face_image_path: Path to the new face image
            user_id: ID of the user
            silent: If True, don't show training messages
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract face encoding
            face_encoding = self.extract_face_encoding(face_image_path)
            if face_encoding is None:
                logger.error("Could not extract face encoding from new image")
                return False
                
            # Add to training data
            self.face_encodings.append(face_encoding)
            self.face_labels.append(user_id)
            
            # Retrain model
            if not silent:
                logger.info("Retraining model with new data...")
                
            success = self.train_face_recognition_model()
            
            if success and not silent:
                logger.info("Model updated successfully")
                
            return success
            
        except Exception as e:
            logger.error(f"Error updating model with new data: {e}")
            return False
    
    def recognize_face(self, face_image_path, confidence_threshold=0.6):
        """
        Recognize a face in an image
        
        Args:
            face_image_path: Path to the face image
            confidence_threshold: Minimum confidence score to consider a match
            
        Returns:
            User ID if recognized, None otherwise
        """
        try:
            # Extract face encoding
            face_encoding = self.extract_face_encoding(face_image_path)
            
            if face_encoding is None:
                return None
                
            # Predict user ID
            user_id = self.model.predict([face_encoding])[0]
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba([face_encoding])[0]
            confidence = max(probabilities)
            
            if confidence < confidence_threshold:
                logger.warning(f"Low confidence ({confidence:.4f}) for user {user_id}")
                return None
                
            logger.info(f"Recognized user {user_id} with confidence {confidence:.4f}")
            return user_id
            
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize AI system
    ai = VeriFaceAI()
    
    # Collect training data
    ai.collect_training_data()
    
    # Train face recognition model
    ai.train_face_recognition_model()
    
    # Train attendance prediction model
    ai.train_attendance_prediction_model()
    
    # Example: Predict attendance for a user
    user_id = "1"  # Replace with actual user ID
    day_of_week = 0  # Monday
    hour_of_day = 9  # 9 AM
    
    predicted_duration = ai.predict_attendance(user_id, day_of_week, hour_of_day)
    if predicted_duration is not None:
        print(f"Predicted attendance duration: {predicted_duration:.2f} minutes") 