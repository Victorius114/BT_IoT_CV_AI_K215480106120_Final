import sqlite3
import pandas as pd

# Đường dẫn database SQLite
db_path = r"E:\Ky2_2024_2025\BT_IoT_CV\db_mysql\Nhan_dien.db"

# Kết nối SQLite
conn = sqlite3.connect(db_path)

# Đọc file CSV
df = pd.read_csv(r"E:\Ky2_2024_2025\BT_IoT_CV\db_mysql\Diem_danh.csv")

# Nhập vào bảng SQLite (nếu chưa có bảng thì SQLite sẽ tạo mới)
df.to_sql("Diem_danh", conn, if_exists="replace", index=False)

# Đóng kết nối
conn.close()

print("Dữ liệu đã nhập vào SQLite thành công!")
