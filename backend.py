import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity

class Backend:
    def __init__(self):
        self.conn = sqlite3.connect('users.db', check_same_thread=False)
        self.c = self.conn.cursor()
        self.setup_database()
        self.load_menu_data()

    def setup_database(self):
        """Initialize database tables"""
        # Create users table
        self.c.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT NOT NULL,
                          phone TEXT,
                          dob TEXT,
                          face_encoding BLOB,
                          preferences TEXT)''')
        
        # Create orders table
        self.c.execute('''CREATE TABLE IF NOT EXISTS orders
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id INTEGER,
                          items TEXT,
                          order_date DATETIME,
                          total_amount REAL,
                          FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        self.conn.commit()

    def load_menu_data(self):
        """Load and process menu data"""
        try:
            self.menu_data = pd.read_csv("D:/ai project gp robo/aimlproject-main/indian_food.csv")
            self.process_menu_data()
        except Exception as e:
            print(f"Error loading menu data: {e}")
            self.menu_data = None

    def process_menu_data(self):
        """Process menu data for recommendations"""
        if self.menu_data is not None:
            # Create feature vectors for recommendations
            self.menu_data['Features'] = self.menu_data['Ingredients'].fillna('') + ' ' + \
                                       self.menu_data['Course'].fillna('') + ' ' + \
                                       self.menu_data['Flavour'].fillna('')
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = tfidf.fit_transform(self.menu_data['Features'])

    def verify_face(self, face_encoding):
        """Verify face against database and find best match"""
        try:
            # Get all users with face encodings
            self.c.execute("SELECT id, name, face_encoding FROM users")
            users = self.c.fetchall()
            
            print(f"Comparing with {len(users)} stored faces")
            
            best_match = None
            best_distance = float('inf')
            
            for user_id, name, stored_encoding_blob in users:
                if stored_encoding_blob is None:
                    continue
                    
                # Convert blob back to numpy array
                stored_encoding = np.frombuffer(stored_encoding_blob, dtype=np.float64)
                
                # Calculate face distance
                face_distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
                confidence = (1 - face_distance) * 100  # Convert distance to confidence percentage
                
                # Check if this is the best match so far
                if face_distance < best_distance and face_distance < 0.6:  # 0.6 is our threshold
                    best_distance = face_distance
                    best_match = (user_id, name, confidence)
            
            if best_match:
                user_id, name, confidence = best_match
                print(f"Found match: {name} (ID: {user_id}) with confidence: {confidence:.2f}%")
                
                # Only return match if confidence is above 50%
                if confidence >= 50:
                    print(f"Match accepted - confidence above threshold")
                    return user_id, name
                else:
                    print(f"Match rejected - confidence too low ({confidence:.2f}%)")
                    return None, None
            
            print("No matching face found in database")
            return None, None
            
        except Exception as e:
            print(f"Error in face verification: {e}")
            return None, None

    def register_user(self, name, phone, dob, face_encoding):
        """Register new user with face encoding"""
        try:
            # Convert numpy array to bytes for storage
            encoding_bytes = face_encoding.tobytes()
            
            # Insert user data
            self.c.execute("""
                INSERT INTO users (name, phone, dob, face_encoding)
                VALUES (?, ?, ?, ?)
            """, (name, phone, dob, encoding_bytes))
            
            self.conn.commit()
            new_user_id = self.c.lastrowid
            print(f"Successfully registered user {name} with ID {new_user_id}")
            return new_user_id
            
        except Exception as e:
            print(f"Error registering user: {e}")
            return None

    def get_recommendations(self, user_id, n_recommendations=5):
        """Get food recommendations based on user history"""
        try:
            # Get user's past orders
            self.c.execute("SELECT items FROM orders WHERE user_id = ? ORDER BY order_date DESC", (user_id,))
            orders_result = self.c.fetchone()
            
            if not orders_result:
                # If no orders found, return random recommendations
                return self.menu_data.sample(n_recommendations)['Name'].tolist()
            
            # Convert orders string to list
            orders = orders_result[0].split(',') if orders_result[0] else []
            
            if not orders:
                return self.menu_data.sample(n_recommendations)['Name'].tolist()
            
            # Get indices of previously ordered items
            ordered_indices = []
            for order in orders:
                idx = self.menu_data[self.menu_data['Name'] == order.strip()].index
                if not idx.empty:
                    ordered_indices.append(idx[0])
            
            if ordered_indices:
                # Convert to numpy array and reshape
                ordered_features = np.asarray(self.tfidf_matrix[ordered_indices].mean(axis=0)).flatten()
                
                # Calculate similarity scores
                sim_scores = cosine_similarity(
                    ordered_features.reshape(1, -1),
                    self.tfidf_matrix.toarray()
                ).flatten()
                
                # Get top N similar items
                sim_indices = sim_scores.argsort()[-n_recommendations:][::-1]
                return self.menu_data.iloc[sim_indices]['Name'].tolist()
            
            return self.menu_data.sample(n_recommendations)['Name'].tolist()
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return self.menu_data.sample(n_recommendations)['Name'].tolist()

    def save_order(self, user_id, items, total_amount):
        """Save a new order"""
        try:
            items_str = ','.join(items)
            self.c.execute("""INSERT INTO orders (user_id, items, order_date, total_amount) 
                            VALUES (?, ?, ?, ?)""",
                            (user_id, items_str, datetime.now(), total_amount))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving order: {e}")
            return False

    def get_user_orders(self, user_id):
        """Get user's order history"""
        try:
            self.c.execute("""SELECT items, order_date, total_amount 
                            FROM orders WHERE user_id = ? 
                            ORDER BY order_date DESC""", (user_id,))
            return self.c.fetchall()
        except Exception as e:
            print(f"Error fetching orders: {e}")
            return []

    def update_user_preferences(self, user_id, preferences):
        """Update user preferences"""
        try:
            preferences_str = ','.join(preferences)
            self.c.execute("UPDATE users SET preferences = ? WHERE id = ?", 
                          (preferences_str, user_id))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating preferences: {e}")
            return False

    def close(self):
        """Close database connection"""
        self.conn.close()

    def get_user_details(self, user_id):
        """Get user details from database"""
        try:
            self.c.execute("""SELECT name, phone, dob 
                             FROM users 
                             WHERE id = ?""", (user_id,))
            result = self.c.fetchone()
            if result:
                return {
                    'name': result[0],
                    'phone': result[1],
                    'dob': result[2]
                }
            return None
        except Exception as e:
            print(f"Error fetching user details: {e}")
            return {
                'name': 'Unknown',
                'phone': 'Not available',
                'dob': 'Not available'
            }

    def get_user_name(self, user_id):
        """Get user name"""
        try:
            self.c.execute("SELECT name FROM users WHERE id = ?", (user_id,))
            result = self.c.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Error fetching user name: {e}")
            return None 