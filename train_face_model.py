import os
import sqlite3
import numpy as np
import pickle
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from datetime import datetime
import sys

# Add the current directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("known_faces", exist_ok=True)
os.makedirs("face_data", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaceTraining")

def load_face_encodings_from_db():
    """
    Load face encodings and labels from the database.
    Returns a tuple of (encodings, labels).
    """
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Get all users with their encodings
        cursor.execute("SELECT id, name, encoding FROM users")
        users = cursor.fetchall()
        
        if not users:
            logger.warning("No users found in the database.")
            return None, None
        
        encodings = []
        labels = []
        
        for user_id, name, encoding_blob in users:
            if encoding_blob is None:
                logger.warning(f"User {name} (ID: {user_id}) has no face encoding.")
                continue
                
            try:
                # Convert blob to numpy array
                encoding = np.frombuffer(encoding_blob, dtype=np.float32)
                
                # Check if encoding is valid (has the expected shape)
                if encoding.shape[0] == 0:
                    logger.warning(f"User {name} (ID: {user_id}) has an empty face encoding.")
                    continue
                    
                encodings.append(encoding)
                labels.append(user_id)  # Use user_id as the label
            except Exception as e:
                logger.error(f"Error processing encoding for user {name} (ID: {user_id}): {str(e)}")
                continue
        
        conn.close()
        
        if not encodings:
            logger.warning("No valid face encodings found.")
            return None, None
            
        return np.array(encodings), np.array(labels)
        
    except Exception as e:
        logger.error(f"Error loading face encodings from database: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def train_face_classifier(encodings, labels, classifier_type='svm'):
    """
    Train a face classifier using the provided encodings and labels.
    
    Args:
        encodings: numpy array of face encodings
        labels: numpy array of user IDs
        classifier_type: 'svm' or 'knn'
        
    Returns:
        Trained classifier
    """
    if encodings is None or labels is None or len(encodings) == 0:
        logger.error("No valid encodings or labels provided for training.")
        return None
        
    try:
        # Check if we have enough samples for training
        if len(encodings) < 2:
            logger.error(f"Not enough samples for training. Found only {len(encodings)} samples.")
            return None
            
        # Check if we have enough unique classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            logger.error(f"Not enough unique classes for training. Found only {len(unique_labels)} class(es). Need at least 2 users to train the model.")
            return None
            
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            encodings, labels, test_size=0.2, random_state=42
        )
        
        # Initialize classifier
        if classifier_type.lower() == 'svm':
            classifier = SVC(kernel='linear', probability=True)
            logger.info("Using SVM classifier")
        else:  # Default to KNN
            classifier = KNeighborsClassifier(n_neighbors=3)
            logger.info("Using KNN classifier")
        
        # Train the classifier
        logger.info(f"Training {classifier_type.upper()} classifier with {len(X_train)} samples...")
        classifier.fit(X_train, y_train)
        
        # Evaluate the classifier
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Classifier accuracy: {accuracy:.2f}")
        
        return classifier
        
    except Exception as e:
        logger.error(f"Error training classifier: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def save_classifier(classifier, classifier_type='svm'):
    """
    Save the trained classifier to a file.
    
    Args:
        classifier: Trained classifier
        classifier_type: 'svm' or 'knn'
        
    Returns:
        Path to the saved model file
    """
    if classifier is None:
        logger.error("No classifier to save.")
        return None
        
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"face_recognition_{classifier_type}_{timestamp}.pkl"
        model_path = os.path.join("models", model_filename)
        
        # Save the classifier
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
            
        logger.info(f"Classifier saved to {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Error saving classifier: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def save_encodings_and_labels(encodings, labels):
    """
    Save encodings and labels to a file for future use.
    
    Args:
        encodings: numpy array of face encodings
        labels: numpy array of user IDs
        
    Returns:
        Path to the saved data file
    """
    if encodings is None or labels is None:
        logger.error("No encodings or labels to save.")
        return None
        
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = os.path.join("face_data", f"face_encodings_{timestamp}.pkl")
        
        # Save the data
        data = {
            'encodings': encodings,
            'labels': labels
        }
        
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Face encodings and labels saved to {data_path}")
        return data_path
        
    except Exception as e:
        logger.error(f"Error saving encodings and labels: {str(e)}")
        return None

def train_and_save_face_model(classifier_type='svm'):
    """
    Train and save the face recognition model.
    
    Args:
        classifier_type (str): Type of classifier to use ('svm' or 'knn')
        
    Returns:
        dict: Dictionary containing success status and message
    """
    try:
        # Load face encodings from database
        encodings, labels = load_face_encodings_from_db()
        
        if encodings is None or labels is None:
            return {
                'success': False,
                'message': 'Failed to load face encodings from database'
            }
            
        if len(encodings) < 2:
            return {
                'success': False,
                'message': 'Not enough face encodings to train the model. At least 2 users with face encodings are required.'
            }
            
        # Train the classifier
        classifier = train_face_classifier(encodings, labels, classifier_type)
        if classifier is None:
            return {
                'success': False,
                'message': 'Failed to train the classifier'
            }
            
        # Save the classifier
        if not save_classifier(classifier, classifier_type):
            return {
                'success': False,
                'message': 'Failed to save the classifier'
            }
            
        return {
            'success': True,
            'message': 'Model trained and saved successfully'
        }
        
    except Exception as e:
        logger.error(f"Error in train_and_save_face_model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'message': f'Error during model training: {str(e)}'
        }

if __name__ == "__main__":
    # Train and save the face model
    result = train_and_save_face_model(classifier_type='svm')
    print(result) 