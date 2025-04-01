import sqlite3
import os

def setup_inventory_database():
    conn = sqlite3.connect('jewelry_inventory.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            product_image BLOB,
            price REAL NOT NULL,
            quantity INTEGER NOT NULL
        )
    """)
    
    cursor.execute("SELECT COUNT(*) FROM inventory")
    count = cursor.fetchone()[0]
    print(f"Existing records in inventory: {count}")
    
    if count == 0:
        sample_items = [
            ('J001', 'Gold Necklace', 'jewel1.jpg', 599.99, 5),
            ('J002', 'Diamond Ring', 'jewel2.jpg', 1299.99, 3),
            ('J003', 'Silver Bracelet', 'jewel3.jpg', 199.99, 8),
            ('J004', 'Pearl Earrings', 'jewel4.jpg', 149.99, 6),
            ('J005', 'Emerald Pendant', 'jewel5.jpg', 799.99, 4),
            ('J006', 'Ruby Ring', 'jewel6.jpg', 999.99, 2),
            ('J007', 'Sapphire Bracelet', 'jewel7.jpg', 399.99, 7),
            ('J008', 'Gold Chain', 'jewel8.jpg', 349.99, 5),
            ('J009', 'Diamond Studs', 'jewel9.jpg', 699.99, 3),
            ('J010', 'Silver Anklet', 'jewel10.jpg', 99.99, 10)
        ]
        
        print(f"Current working directory: {os.getcwd()}")
        for product_id, product_name, image_file, price, quantity in sample_items:
            image_data = None
            if os.path.exists(image_file):
                try:
                    with open(image_file, 'rb') as f:
                        image_data = f.read()
                    print(f"Loaded {image_file} successfully, size: {len(image_data)} bytes")
                except Exception as e:
                    print(f"Error loading image {image_file}: {e}")
                    image_data = None
            else:
                print(f"Image file {image_file} not found")
            
            cursor.execute("""
                INSERT INTO inventory (product_id, product_name, product_image, price, quantity)
                VALUES (?, ?, ?, ?, ?)
            """, (product_id, product_name, image_data, price, quantity))
        print("Initial inventory data inserted.")
    
    conn.commit()
    
    # Always display current inventory status
    cursor.execute("SELECT product_id, product_name, length(product_image), price, quantity FROM inventory")
    print("\nCurrent Inventory Status:")
    for row in cursor.fetchall():
        product_id, product_name, image_size, price, quantity = row
        print(f"Product {product_id} - {product_name}: Price = ${price:.2f}, Quantity = {quantity}, Image size = {image_size if image_size else 'None'} bytes")
    
    conn.close()

if __name__ == "__main__":
    setup_inventory_database()