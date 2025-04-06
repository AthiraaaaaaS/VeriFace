import os
import cv2
import numpy as np
from datetime import datetime
import sqlite3
import logging
from ai_training import VeriFaceAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VeriFace_AI_Integration')

class VeriFaceAIIntegration:
    def __init__(self, db_path='attendance.db', known_faces_dir='known_faces'):
        """
        Initialize the VeriFace AI integration
        
        Args:
            db_path: Path to the SQLite database
            known_faces_dir: Directory containing known face images
        """
        self.db_path = db_path
        self.known_faces_dir = known_faces_dir
        self.ai = VeriFaceAI(db_path=db_path, known_faces_dir=known_faces_dir)
        
        # Create directories if they don't exist
        os.makedirs(known_faces_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Initialize AI system
        self.initialize_ai()
        
    def initialize_ai(self):
        """Initialize the AI system by collecting data and training models"""
        logger.info("Initializing VeriFace AI system...")
        
        # Collect training data
        self.ai.collect_training_data()
        
        # Train face recognition model
        self.ai.train_face_recognition_model()
        
        # Train attendance prediction model
        self.ai.train_attendance_prediction_model()
        
        logger.info("VeriFace AI system initialized successfully")
    
    def register_user_with_ai(self, name, email, phone, image_path):
        """
        Register a new user with AI training
        
        Args:
            name: User's name
            email: User's email
            phone: User's phone number
            image_path: Path to the user's face image
            
        Returns:
            User ID if successful, None otherwise
        """
        try:
            # Clean the name to create a valid filename
            clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            saved_image_path = os.path.join(self.known_faces_dir, f"{clean_name}.jpg")
            
            # Check if file already exists
            if os.path.exists(saved_image_path):
                logger.warning(f"Image for {name} already exists. Replacing it.")
            
            # Save image
            cv2.imwrite(saved_image_path, cv2.imread(image_path))
            
            # Get face encoding
            face_encoding = self.ai.extract_face_encoding(saved_image_path)
            if face_encoding is None:
                logger.error("Could not detect face in the image")
                return None
                
            face_encoding_blob = face_encoding.astype(np.float32).tobytes()
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create users table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE,
                    encoding BLOB,
                    phone TEXT,
                    email TEXT
                )
            """)
            
            # Check if user already exists
            cursor.execute("SELECT user_id FROM users WHERE name = ?", (name,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                # Update existing user
                user_id = existing_user[0]
                cursor.execute("""
                    UPDATE users 
                    SET encoding = ?, phone = ?, email = ?, image_path = ?
                    WHERE name = ?
                """, (face_encoding_blob, phone, email, saved_image_path, name))
            else:
                # Insert new user
                cursor.execute("""
                    INSERT INTO users (name, encoding, phone, email, image_path)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, face_encoding_blob, phone, email, saved_image_path))
                user_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            # Update AI model with new data 
            self.ai.update_model_with_new_data(saved_image_path, str(user_id), silent=True)
            
            logger.info(f"User {name} registered successfully")
            return user_id
            
        except Exception as e:
            logger.error(f"Error registering user with AI: {e}")
            return None
    
    def recognize_user(self, image_path, confidence_threshold=0.6):
        """
        Recognize a user from an image
        
        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence score to consider a match
            
        Returns:
            User ID if recognized, None otherwise
        """
        try:
            # Recognize face using AI
            user_id = self.ai.recognize_face(image_path, confidence_threshold)
            
            if user_id is None:
                logger.warning("Face not recognized")
                return None
                
            # Get user details from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT user_id, name FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            
            conn.close()
            
            if user is None:
                logger.warning(f"User with ID {user_id} not found in database")
                return None
                
            logger.info(f"Recognized user: {user[1]} (ID: {user[0]})")
            return user[0]
            
        except Exception as e:
            logger.error(f"Error recognizing user: {e}")
            return None
    
    def record_attendance(self, user_id):
        """
        Record attendance for a user
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if attendance already recorded for today
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("SELECT user_id, name FROM users WHERE user_id = ?", (str(user_id),))
            
            existing_record = cursor.fetchone()
            
            if existing_record:
                # Update last_seen time
                cursor.execute("""
                    UPDATE attendance 
                    SET last_seen = ? 
                    WHERE user_id = ?
                """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), existing_record[0]))
                
                logger.info(f"Updated attendance for user {user_id}")
            else:
                # Create new attendance record
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("""
                    INSERT INTO attendance (user_id, first_seen, last_seen)
                    VALUES (?, ?, ?)
                """, (user_id, now, now))
                
                logger.info(f"Created new attendance record for user {user_id}")
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording attendance: {e}")
            return False
    
    def predict_user_attendance(self, user_id):
        """
        Predict attendance for a user today
        
        Args:
            user_id: User ID
            
        Returns:
            Predicted duration in minutes or None if prediction failed
        """
        try:
            # Get current day and hour
            now = datetime.now()
            day_of_week = now.weekday()  # 0 = Monday, 6 = Sunday
            hour_of_day = now.hour
            
            # Predict attendance
            predicted_duration = self.ai.predict_attendance(user_id, day_of_week, hour_of_day)
            
            if predicted_duration is not None:
                logger.info(f"Predicted attendance duration for user {user_id}: {predicted_duration:.2f} minutes")
                return predicted_duration
            else:
                logger.warning(f"Could not predict attendance for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error predicting attendance: {e}")
            return None
    
    def get_attendance_analytics(self, user_id=None, days=30):
        """
        Get attendance analytics
        
        Args:
            user_id: User ID (if None, get analytics for all users)
            days: Number of days to analyze
            
        Returns:
            Dictionary with analytics data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get date range
            end_date = datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # Build query
            if user_id is not None:
                query = """
                    SELECT users.name, attendance.first_seen, attendance.last_seen
                    FROM attendance
                    JOIN users ON attendance.user_id = users.user_id
                    WHERE users.user_id = ? AND attendance.first_seen >= ?
                    ORDER BY attendance.first_seen
                """
                cursor.execute(query, (user_id, start_date.strftime("%Y-%m-%d")))
            else:
                query = """
                    SELECT users.name, attendance.first_seen, attendance.last_seen
                    FROM attendance
                    JOIN users ON attendance.user_id = users.user_id
                    WHERE attendance.first_seen >= ?
                    ORDER BY attendance.first_seen
                """
                cursor.execute(query, (start_date.strftime("%Y-%m-%d"),))
            
            records = cursor.fetchall()
            
            # Process records
            analytics = {
                'total_attendance_days': 0,
                'average_duration': 0,
                'attendance_by_day': {},
                'attendance_by_hour': {},
                'records': []
            }
            
            total_duration = 0
            
            for name, first_seen, last_seen in records:
                # Convert string timestamps to datetime objects
                first_seen_dt = datetime.strptime(first_seen, "%Y-%m-%d %H:%M:%S")
                last_seen_dt = datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S")
                
                # Calculate duration in minutes
                duration = (last_seen_dt - first_seen_dt).total_seconds() / 60
                
                # Extract day of week (0 = Monday, 6 = Sunday)
                day_of_week = first_seen_dt.weekday()
                day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]
                
                # Extract hour of day (0-23)
                hour_of_day = first_seen_dt.hour
                
                # Update analytics
                analytics['total_attendance_days'] += 1
                total_duration += duration
                
                # Update day statistics
                if day_name not in analytics['attendance_by_day']:
                    analytics['attendance_by_day'][day_name] = 0
                analytics['attendance_by_day'][day_name] += 1
                
                # Update hour statistics
                if hour_of_day not in analytics['attendance_by_hour']:
                    analytics['attendance_by_hour'][hour_of_day] = 0
                analytics['attendance_by_hour'][hour_of_day] += 1
                
                # Add record
                analytics['records'].append({
                    'name': name,
                    'date': first_seen_dt.date().strftime("%Y-%m-%d"),
                    'day': day_name,
                    'first_seen': first_seen,
                    'last_seen': last_seen,
                    'duration': duration
                })
            
            # Calculate average duration
            if analytics['total_attendance_days'] > 0:
                analytics['average_duration'] = total_duration / analytics['total_attendance_days']
            
            conn.close()
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting attendance analytics: {e}")
            return None
    
    def retrain_models(self):
        """
        Retrain all AI models with latest data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Retraining AI models...")
            
            # Collect latest data
            self.ai.collect_training_data()
            
            # Retrain face recognition model
            face_recognition_success = self.ai.train_face_recognition_model()
            
            # Retrain attendance prediction model
            attendance_prediction_success = self.ai.train_attendance_prediction_model()
            
            if face_recognition_success and attendance_prediction_success:
                logger.info("AI models retrained successfully")
                return True
            else:
                logger.warning("Some models failed to retrain")
                return False
                
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize integration
    integration = VeriFaceAIIntegration()
    
    # Example: Register a new user
    # user_id = integration.register_user_with_ai("John Doe", "john@example.com", "1234567890", "path/to/image.jpg")
    
    # Example: Recognize a user
    # recognized_user_id = integration.recognize_user("path/to/image.jpg")
    
    # Example: Record attendance
    # if recognized_user_id:
    #     integration.record_attendance(recognized_user_id)
    
    # Example: Get attendance analytics
    # analytics = integration.get_attendance_analytics()
    # if analytics:
    #     print(f"Total attendance days: {analytics['total_attendance_days']}")
    #     print(f"Average duration: {analytics['average_duration']:.2f} minutes") 