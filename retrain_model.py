import os
import sqlite3
import numpy as np
from sklearn.svm import SVC
import pickle
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrain_model():
    """
    Retrain the face recognition model with optimized parameters.
    """
    try:
        # Load face encodings from database
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Get all users with their encodings
        cursor.execute("SELECT user_id, name, encoding FROM users")
        users = cursor.fetchall()
        
        if not users:
            logger.error("No users found in database!")
            return False
            
        encodings = []
        labels = []
        names = {}
        
        for user_id, name, encoding_blob in users:
            if encoding_blob is None:
                logger.warning(f"User {name} (ID: {user_id}) has no face encoding.")
                continue
                
            try:
                encoding = np.frombuffer(encoding_blob, dtype=np.float32)
                encodings.append(encoding)
                labels.append(user_id)
                names[user_id] = name
                logger.info(f"Loaded encoding for {name} (ID: {user_id})")
            except Exception as e:
                logger.error(f"Error processing encoding for {name}: {e}")
                continue
        
        if len(encodings) < 2:
            logger.error("Need at least 2 users with valid encodings!")
            return False
        
        # Convert to numpy arrays
        X = np.array(encodings)
        y = np.array(labels)
        
        # Train SVM with optimized parameters
        classifier = SVC(
            kernel='rbf',  # RBF kernel often works better for face recognition
            probability=True,
            C=10.0,  # Increased C for stricter classification
            gamma='scale',
            random_state=42
        )
        
        logger.info("Training model...")
        classifier.fit(X, y)
        
        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join("models", f"face_recognition_svm_{timestamp}.pkl")
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Delete old model files
        for file in os.listdir("models"):
            if file.endswith(".pkl"):
                try:
                    os.remove(os.path.join("models", file))
                    logger.info(f"Deleted old model: {file}")
                except Exception as e:
                    logger.error(f"Could not delete {file}: {e}")
        
        # Save new model
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Trained with {len(encodings)} face encodings")
        logger.info("Users in model:")
        for user_id, name in names.items():
            logger.info(f"  ID: {user_id}, Name: {name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False
    finally:
        try:
            conn.close()
        except:
            pass

if __name__ == "__main__":
    print("This will retrain the face recognition model with optimized parameters.")
    response = input("Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        if retrain_model():
            print("\n✅ Model retrained successfully!")
            print("\nNext steps:")
            print("1. Restart the attendance system")
            print("2. Test face recognition")
        else:
            print("\n❌ Error retraining model. Check the logs above.")
    else:
        print("Operation cancelled.") 