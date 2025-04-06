import sqlite3
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_and_fix_database():
    """
    Verify database integrity and fix any issues.
    """
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Check tables
        logger.info("Checking database tables...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"Found tables: {[table[0] for table in tables]}")
        
        # Check users table
        logger.info("\nChecking users table...")
        cursor.execute("SELECT user_id, name, email, phone FROM users")
        users = cursor.fetchall()
        logger.info(f"Found {len(users)} users:")
        for user in users:
            logger.info(f"ID: {user[0]}, Name: {user[1]}, Email: {user[2]}, Phone: {user[3]}")
        
        # Check attendance table
        logger.info("\nChecking attendance table...")
        cursor.execute("SELECT user_id, first_seen, last_seen FROM attendance")
        attendance = cursor.fetchall()
        logger.info(f"Found {len(attendance)} attendance records:")
        for record in attendance:
            logger.info(f"User ID: {record[0]}, First seen: {record[1]}, Last seen: {record[2]}")
        
        # Check for orphaned attendance records (records with no corresponding user)
        logger.info("\nChecking for orphaned attendance records...")
        cursor.execute("""
            SELECT a.user_id, a.first_seen, a.last_seen 
            FROM attendance a 
            LEFT JOIN users u ON a.user_id = u.user_id 
            WHERE u.user_id IS NULL
        """)
        orphaned = cursor.fetchall()
        if orphaned:
            logger.warning(f"Found {len(orphaned)} orphaned attendance records:")
            for record in orphaned:
                logger.warning(f"Orphaned record - User ID: {record[0]}, First seen: {record[1]}")
                
            # Delete orphaned records
            cursor.execute("""
                DELETE FROM attendance 
                WHERE user_id IN (
                    SELECT a.user_id 
                    FROM attendance a 
                    LEFT JOIN users u ON a.user_id = u.user_id 
                    WHERE u.user_id IS NULL
                )
            """)
            conn.commit()
            logger.info("Deleted orphaned attendance records")
        else:
            logger.info("No orphaned attendance records found")
        
        conn.close()
        logger.info("\nDatabase verification complete")
        
    except Exception as e:
        logger.error(f"Error verifying database: {e}")
        return False
    
    return True

if __name__ == "__main__":
    verify_and_fix_database() 