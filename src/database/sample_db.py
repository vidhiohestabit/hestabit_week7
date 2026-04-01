import sqlite3
import os

os.makedirs("database", exist_ok=True)

conn = sqlite3.connect("database/sample.db")
cursor = conn.cursor()

# 🔥 Drop old table (clean start)
cursor.execute("DROP TABLE IF EXISTS sales")

# ✅ New improved table
cursor.execute("""
CREATE TABLE sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artist TEXT,
    genre TEXT,
    revenue REAL,
    year INTEGER,
    country TEXT
)
""")

# ✅ More realistic data
data = [
    ("Artist A", "Pop", 100.5, 2023, "USA"),
    ("Artist B", "Rock", 50.2, 2023, "UK"),
    ("Artist C", "Hip-Hop", 120.0, 2023, "USA"),
    ("Artist D", "Jazz", 80.0, 2023, "France"),
    ("Artist E", "Pop", 95.0, 2023, "India"),

    ("Artist A", "Pop", 70.0, 2022, "USA"),
    ("Artist B", "Rock", 65.0, 2022, "UK"),
    ("Artist C", "Hip-Hop", 110.0, 2022, "USA"),
    ("Artist D", "Jazz", 60.0, 2022, "France"),
    ("Artist E", "Pop", 85.0, 2022, "India"),

    ("Artist A", "Pop", 130.0, 2024, "USA"),
    ("Artist B", "Rock", 75.0, 2024, "UK"),
    ("Artist C", "Hip-Hop", 140.0, 2024, "USA"),
    ("Artist D", "Jazz", 90.0, 2024, "France"),
    ("Artist E", "Pop", 105.0, 2024, "India"),

    ("Artist F", "Classical", 60.0, 2023, "Germany"),
    ("Artist G", "EDM", 150.0, 2023, "Netherlands"),
    ("Artist H", "Pop", 200.0, 2023, "India"),
    ("Artist I", "Rock", 170.0, 2023, "USA"),
    ("Artist J", "Hip-Hop", 190.0, 2023, "Canada")
]

cursor.executemany("""
INSERT INTO sales (artist, genre, revenue, year, country)
VALUES (?, ?, ?, ?, ?)
""", data)

conn.commit()
conn.close()

print("✅ New database created with rich data!")