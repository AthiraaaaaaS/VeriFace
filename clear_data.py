import os
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_all_data():
    """
    Clear all data while keeping table structures:
    1. Delete all records from tables
    2. Reset auto-increment counters
    3. Delete trained models from models directory
    4. Clear known faces directory
    """
    try:
        # 1. Clear database tables
        logger.info("Clearing database tables...")
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Delete all records from tables
        cursor.execute("DELETE FROM attendance")
        cursor.execute("DELETE FROM users")
        
        # Reset auto-increment counters
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='users' OR name='attendance'")
        
        conn.commit()
        
        # Verify tables are empty
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        logger.info(f"Users table cleared: {user_count} records remaining")
        
        cursor.execute("SELECT COUNT(*) FROM attendance")
        attendance_count = cursor.fetchone()[0]
        logger.info(f"Attendance table cleared: {attendance_count} records remaining")
        
        conn.close()
        
        # 2. Delete trained models from models directory
        logger.info("\nDeleting trained models...")
        models_dir = "models"
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".pkl"):
                    try:
                        file_path = os.path.join(models_dir, file)
                        os.remove(file_path)
                        logger.info(f"Deleted model file: {file}")
                    except Exception as e:
                        logger.error(f"Could not delete model file {file}: {e}")
        else:
            os.makedirs(models_dir)
            logger.info("Created models directory")
        
        # Also check for any .pkl files in root directory
        root_model_files = [f for f in os.listdir(".") if f.endswith(".pkl")]
        for model_file in root_model_files:
            try:
                os.remove(model_file)
                logger.info(f"Deleted model file from root: {model_file}")
            except Exception as e:
                logger.error(f"Could not delete model file {model_file}: {e}")
        
        # 3. Clear known faces directory
        logger.info("\nClearing known faces directory...")
        known_faces_dir = "known_faces"
        if os.path.exists(known_faces_dir):
            # Delete all files in known_faces directory
            for file in os.listdir(known_faces_dir):
                try:
                    file_path = os.path.join(known_faces_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        logger.info(f"Deleted face image: {file}")
                except Exception as e:
                    logger.error(f"Could not delete file {file}: {e}")
        else:
            os.makedirs(known_faces_dir)
            logger.info("Created known_faces directory")
        
        logger.info("\nAll data cleared successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error while clearing data: {e}")
        return False

if __name__ == "__main__":
    print("WARNING: This will delete ALL data while keeping table structures!")
    print("This includes:")
    print("- All user records")
    print("- All attendance records")
    print("- All trained models (in models directory and root)")
    print("- All saved face images")
    print("\nThe database tables will remain intact, but empty.")
    
    response = input("\nAre you sure you want to proceed? Type 'YES' in uppercase to confirm: ")
    if response == 'YES':
        if clear_all_data():
            print("\n✅ Data cleared successfully!")
            print("\nNext steps:")
            print("1. Register new users using the registration form")
            print("2. Train the model (you need at least 2 users)")
            print("3. Start taking attendance")
        else:
            print("\n❌ Error occurred while clearing data. Check the logs above.")
    else:
        print("Operation cancelled.") 