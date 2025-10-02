# src/db.py
import sqlite3
import datetime

class DB:
    def __init__(self, path="predictions.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                motor_type TEXT,
                temperature REAL,
                vibration REAL,
                current REAL,
                speed REAL,
                prediction INTEGER,
                prob REAL
            )
        """)
        self.conn.commit()

    def insert(self, motor_type, temperature, vibration, current, speed, prediction, prob):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO predictions (timestamp, motor_type, temperature, vibration, current, speed, prediction, prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.datetime.now().isoformat(), motor_type, temperature, vibration, current, speed, int(prediction), float(prob) if prob is not None else None))
        self.conn.commit()

    def fetch_all(self):
        cur = self.conn.cursor()
        cur.execute("SELECT timestamp, motor_type, temperature, vibration, current, speed, prediction, prob FROM predictions ORDER BY id DESC")
        return cur.fetchall()
