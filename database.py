import sqlite3
import json # To store embeddings as JSON strings

DATABASE_NAME = 'database.db'

def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_filename TEXT UNIQUE NOT NULL,
        embedding TEXT NOT NULL 
    )
    ''')
    conn.commit()
    conn.close()

def add_face(image_filename, embedding):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    try:
        # Convert embedding list/array to JSON string for storage
        embedding_json = json.dumps(embedding)
        cursor.execute("INSERT INTO faces (image_filename, embedding) VALUES (?, ?)",
                       (image_filename, embedding_json))
        conn.commit()
        return True
    except sqlite3.IntegrityError: # For UNIQUE constraint violation
        print(f"Error: Face with filename {image_filename} already exists in DB.")
        return False
    except Exception as e:
        print(f"Database error: {e}")
        return False
    finally:
        conn.close()

def get_all_faces():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT image_filename, embedding FROM faces")
    faces = []
    for row in cursor.fetchall():
        try:
            # Convert JSON string back to list
            embedding = json.loads(row[1])
            faces.append({'image_filename': row[0], 'embedding': embedding})
        except json.JSONDecodeError:
            print(f"Error decoding embedding for {row[0]}")
            continue # Skip this problematic entry
    conn.close()
    return faces

if __name__ == '__main__':
    init_db() # Initialize DB when script is run directly
    print("Database initialized.")