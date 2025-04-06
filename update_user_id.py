import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_user_id(old_id, new_id):
    """
    Update a user's ID in the database and update related attendance records.
    """
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # First, check if new_id already exists
        cursor.execute("SELECT name FROM users WHERE id = ?", (new_id,))
        existing = cursor.fetchone()
        if existing:
            logger.error(f"Cannot update: ID {new_id} is already in use")
            return False
        
        # Start transaction
        cursor.execute("BEGIN TRANSACTION")
        
        try:
            # Get user info before update
            cursor.execute("SELECT name FROM users WHERE id = ?", (old_id,))
            user = cursor.fetchone()
            if not user:
                logger.error(f"User with ID {old_id} not found")
                return False
            
            # Update user ID in users table
            cursor.execute("""
                UPDATE users 
                SET id = ? 
                WHERE id = ?
            """, (new_id, old_id))
            
            # Update attendance records
            cursor.execute("""
                UPDATE attendance 
                SET user_id = ? 
                WHERE user_id = ?
            """, (new_id, old_id))
            
            # Commit transaction
            cursor.execute("COMMIT")
            
            logger.info(f"Successfully updated user ID from {old_id} to {new_id}")
            logger.info(f"Updated user: {user[0]}")
            
            # Verify the update
            cursor.execute("SELECT user_id, name FROM users WHERE id = ?", (new_id,))
            updated = cursor.fetchone()
            if updated:
                logger.info(f"Verified update - New ID: {updated[0]}, Name: {updated[1]}")
            
            return True
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Error during update, rolling back: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Database error: {e}")
        return False
    finally:
        try:
            conn.close()
        except:
            pass

if __name__ == "__main__":
    print("This will update a user's ID in the database.")
    print("WARNING: Make sure to retrain the model after updating IDs!")
    
    old_id = 4  # ID to change from
    new_id = 1  # ID to change to
    
    response = input(f"Update user ID from {old_id} to {new_id}? (yes/no): ")
    
    if response.lower() == 'yes':
        if update_user_id(old_id, new_id):
            print("\n✅ User ID updated successfully!")
            print("\nNext steps:")
            print("1. Run the model retraining script")
            print("2. Restart the attendance system")
        else:
            print("\n❌ Error updating user ID. Check the logs above.")
    else:
        print("Operation cancelled.") 