import sqlite3
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize database and create tables with consistent data types."""
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Enable foreign key support
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create users table with TEXT user_id
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                encoding BLOB,
                phone TEXT,
                email TEXT
            )
        """)
        
        # Create attendance table with TEXT user_id to match users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                user_id TEXT,
                first_seen TEXT,
                last_seen TEXT,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        """)
        
        conn.commit()
        
        # Verify tables were created
        print("\n=== Database Tables Created ===")
        
        # Check users table
        print("\nUsers Table Structure:")
        print("-" * 50)
        cursor.execute("PRAGMA table_info(users)")
        for col in cursor.fetchall():
            print(f"Column: {col[1]:<10} Type: {col[2]:<10} {'Primary Key' if col[5] else ''}")
            
        # Check attendance table
        print("\nAttendance Table Structure:")
        print("-" * 50)
        cursor.execute("PRAGMA table_info(attendance)")
        for col in cursor.fetchall():
            print(f"Column: {col[1]:<10} Type: {col[2]:<10}")
            
        # Show foreign keys
        print("\nForeign Keys:")
        print("-" * 50)
        cursor.execute("PRAGMA foreign_key_list(attendance)")
        for fk in cursor.fetchall():
            print(f"Column: {fk[3]} references users({fk[4]})")
        
        conn.close()
        logger.info("Database initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    print("Database Initialization")
    print("This will create the tables with consistent TEXT user_id fields")
    
    response = input("\nContinue? (yes/no): ")
    
    if response.lower() == 'yes':
        if initialize_database():
            print("\n✅ Database initialized successfully!")
            print("\nChanges made:")
            print("1. users table: user_id is TEXT")
            print("2. attendance table: user_id is TEXT")
            print("3. Foreign key now correctly references users(user_id)")
        else:
            print("\n❌ Error initializing database. Check the logs above.")
    else:
        print("Operation cancelled.") 