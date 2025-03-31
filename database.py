import sqlite3

# Connect to the database
db_path = "crowd.db"  # Make sure this path is correct
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch all records
cursor.execute("SELECT * FROM crowd")
rows = cursor.fetchall()

# Display records
for row in rows:
    print(row)

# Close the connection
conn.close()
