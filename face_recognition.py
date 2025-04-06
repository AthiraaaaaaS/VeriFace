import os
import pickle
import numpy as np
import cv2
import sqlite3
import logging
from datetime import datetime
import sys
from insightface.app import FaceAnalysis

# Add the current directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("known_faces", exist_ok=True)

# Initialize InsightFace app
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaceRecognition")

class FaceRecognizer:
    def __init__(self, model_path=None, confidence_threshold=0.6):
        """
        Initialize the face recognizer.
        
        Args:
            model_path: Path to the trained model file. If None, will try to find the latest model.
            confidence_threshold: Minimum confidence score to consider a match.
        """
        self.confidence_threshold = confidence_threshold
        self.classifier = None
        self.user_ids = []
        self.user_names = {}
        
        # Load the model
        if model_path is None:
            model_path = self._find_latest_model()
            
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.warning("No model found. Please train a model first.")
            
        # Load user information
        self._load_user_info()
        logger.info(f"Initialized with users: {self.user_names}")
        
        # Verify model and database consistency
        self._verify_model_database_consistency()
    
    def _find_latest_model(self):
        """Find the latest trained model in the models directory."""
        models_dir = "models"
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory '{models_dir}' not found.")
            return None
            
        # Look for both old and new model filename patterns
        model_files = [f for f in os.listdir(models_dir) 
                      if (f.startswith("face_classifier_") or f.startswith("face_recognition_")) 
                      and f.endswith(".pkl")]
        
        if not model_files:
            logger.warning("No model files found.")
            return None
            
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
        latest_model = os.path.join(models_dir, model_files[0])
        logger.info(f"Found latest model: {latest_model}")
        return latest_model
    
    def _load_model(self, model_path):
        """Load the trained model from file."""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
                
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
                
            logger.info(f"Model loaded from {model_path}")
            
            # Load user information from database
            if not self._load_user_info():
                logger.warning("Failed to load user information, but model was loaded.")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_user_info(self):
        """Load user IDs and names from the database."""
        try:
            if not os.path.exists("attendance.db"):
                logger.error("Database file 'attendance.db' not found.")
                return False
                
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            
            # Check if users table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            if not cursor.fetchone():
                logger.error("Users table not found in database.")
                conn.close()
                return False
                
            cursor.execute("SELECT user_id, name FROM users")
            users = cursor.fetchall()
            
            if not users:
                logger.warning("No users found in the database.")
                self.user_ids = []
                self.user_names = {}
            else:
                self.user_ids = [user[0] for user in users]
                self.user_names = {user[0]: user[1] for user in users}
                logger.info(f"Loaded information for {len(self.user_ids)} users")
            
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error loading user information: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def extract_face_encoding(self, image_path):
        """
        Extract face encoding from an image file.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Face encoding as numpy array or None if no face detected.
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
                
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
                
            # Check if image is empty
            if img.size == 0:
                logger.error(f"Image is empty: {image_path}")
                return None
                
            # Get face embeddings
            faces = app.get(img)
            if not faces:
                logger.warning(f"No face detected in image: {image_path}")
                return None
                
            # Check if embedding is valid
            embedding = faces[0].embedding
            if embedding is None or embedding.shape[0] == 0:
                logger.error(f"Invalid face embedding extracted from: {image_path}")
                return None
                
            logger.info(f"Successfully extracted face encoding from: {image_path}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting face encoding: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def recognize_faces(self, image_path, max_faces=10):
        """
        Recognize multiple faces in an image.
        
        Args:
            image_path: Path to the image file.
            max_faces: Maximum number of faces to detect.
            
        Returns:
            List of tuples (user_id, user_name, confidence) for each recognized face.
        """
        if self.classifier is None:
            logger.error("No model loaded. Please train a model first.")
            return []
            
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Failed to load image: {image_path}")
                return []
                
            # Detect faces
            faces = app.get(img)
            if not faces:
                logger.warning(f"No faces detected in image: {image_path}")
                return []
                
            # Limit to max_faces
            faces = faces[:max_faces]
            
            results = []
            for face in faces:
                # Get face encoding
                face_encoding = face.embedding
                
                # Reshape for prediction
                face_encoding = face_encoding.reshape(1, -1)
                
                # Predict user ID
                user_id = self.classifier.predict(face_encoding)[0]
                
                # Get confidence score
                if hasattr(self.classifier, 'predict_proba'):
                    confidence = self.classifier.predict_proba(face_encoding).max()
                else:
                    # For KNN, we can't get probability directly
                    # Use a simple distance-based approach
                    distances = self.classifier.kneighbors(face_encoding, return_distance=True)[0][0]
                    confidence = 1.0 / (1.0 + distances[0])  # Convert distance to confidence
                    
                # Check if confidence meets threshold
                if confidence < self.confidence_threshold:
                    logger.info(f"Face recognized but confidence ({confidence:.2f}) below threshold ({self.confidence_threshold})")
                    results.append((None, None, confidence))
                    continue
                    
                # Get user name
                user_name = self.user_names.get(user_id, "Unknown")
                
                logger.info(f"Face recognized as {user_name} (ID: {user_id}) with confidence {confidence:.2f}")
                results.append((user_id, user_name, confidence))
                
            return results
            
        except Exception as e:
            logger.error(f"Error recognizing faces: {str(e)}")
            return []
    
    def recognize_face(self, image_path):
        """
        Recognize a face in the given image (maintained for backward compatibility).
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Tuple of (user_id, user_name, confidence) if recognized, (None, None, 0) otherwise.
        """
        results = self.recognize_faces(image_path, max_faces=1)
        if results:
            return results[0]
        return None, None, 0
    
    def recognize_face_from_frame(self, frame):
        """
        Recognize a face in the given video frame.
        
        Args:
            frame: OpenCV image frame.
            
        Returns:
            Tuple of (user_id, user_name, confidence) if recognized, (None, None, 0) otherwise.
        """
        if self.classifier is None:
            logger.error("No model loaded. Please train a model first.")
            return None, None, 0
            
        try:
            # Extract face encoding directly from frame
            faces = app.get(frame)
            if not faces:
                return None, None, 0
                
            face_encoding = faces[0].embedding
            
            # Reshape for prediction
            face_encoding = face_encoding.reshape(1, -1)
            
            # Predict user ID
            user_id = self.classifier.predict(face_encoding)[0]
            
            # Get confidence score
            if hasattr(self.classifier, 'predict_proba'):
                confidence = self.classifier.predict_proba(face_encoding).max()
            else:
                # For KNN, we can't get probability directly
                # Use a simple distance-based approach
                distances = self.classifier.kneighbors(face_encoding, return_distance=True)[0][0]
                confidence = 1.0 / (1.0 + distances[0])  # Convert distance to confidence
                
            # Check if confidence meets threshold
            if confidence < self.confidence_threshold:
                return None, None, confidence
                
            # Get user name
            user_name = self.user_names.get(user_id, "Unknown")
            
            return user_id, user_name, confidence
            
        except Exception as e:
            logger.error(f"Error recognizing face from frame: {str(e)}")
            return None, None, 0
    
    def _verify_model_database_consistency(self):
        """Verify that the model's user IDs match the database."""
        if self.classifier is None:
            return
            
        try:
            # Get unique labels that the model knows about
            if hasattr(self.classifier, 'classes_'):
                model_user_ids = set(self.classifier.classes_)
            else:
                logger.warning("Classifier doesn't have classes_ attribute")
                return
                
            # Get user IDs from database
            db_user_ids = set(self.user_ids)
            
            # Check for mismatches
            missing_in_db = model_user_ids - db_user_ids
            if missing_in_db:
                logger.warning(f"Model contains user IDs not in database: {missing_in_db}")
                logger.warning("Retraining model to ensure consistency...")
                self._retrain_model()
                
        except Exception as e:
            logger.error(f"Error verifying model-database consistency: {e}")
    
    def _retrain_model(self):
        """Retrain the model using current database data."""
        try:
            from train_face_model import train_and_save_face_model
            result = train_and_save_face_model(classifier_type='svm')
            if result['success']:
                logger.info("Model retrained successfully")
                # Reload the model
                model_path = self._find_latest_model()
                if model_path and os.path.exists(model_path):
                    self._load_model(model_path)
            else:
                logger.error(f"Model retraining failed: {result['message']}")
        except Exception as e:
            logger.error(f"Error retraining model: {e}")

    def record_attendance(self, user_id):
        """Record attendance for a user."""
        if user_id is None:
            logger.error("Cannot record attendance for None user_id")
            return False
        
        try:
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            
            # First verify if the user exists in users table
            cursor.execute("SELECT user_id, name FROM users WHERE user_id = ?", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                logger.error(f"User ID {user_id} not found in users table")
                return False
            
            logger.info(f"Recording attendance for user: {user[1]} (ID: {user[0]})")
            
            # Get current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            today = datetime.now().strftime("%Y-%m-%d")
            
            try:
                # Check if attendance already recorded today
                cursor.execute("""
                    SELECT user_id, first_seen, last_seen 
                    FROM attendance 
                    WHERE user_id = ? AND DATE(first_seen) = ?
                """, (user_id, today))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update last_seen
                    cursor.execute("""
                        UPDATE attendance 
                        SET last_seen = ? 
                        WHERE user_id = ?
                    """, (current_time, user_id))
                    logger.info(f"Updated last_seen for user {user[1]} (ID: {user[0]})")
                else:
                    # Insert new attendance record
                    cursor.execute("""
                        INSERT INTO attendance (user_id, first_seen, last_seen)
                        VALUES (?, ?, ?)
                    """, (user_id, current_time, current_time))
                    logger.info(f"Created new attendance record for user {user[1]} (ID: {user[0]})")
                
                conn.commit()
                return True
                
            except sqlite3.Error as e:
                logger.error(f"Database error while recording attendance: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error recording attendance: {e}")
            return False
        finally:
            try:
                conn.close()
            except:
                pass

# Example usage
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    image_path = "known_faces/sample.jpg"
    if os.path.exists(image_path):
        user_id, user_name, confidence = recognizer.recognize_face(image_path)
        if user_id:
            print(f"Recognized as {user_name} (ID: {user_id}) with confidence {confidence:.2f}")
            recognizer.record_attendance(user_id)
        else:
            print("No face recognized")
    else:
        print(f"Sample image not found: {image_path}") 