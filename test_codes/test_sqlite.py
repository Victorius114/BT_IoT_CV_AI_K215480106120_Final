import sqlite3

# 🔹 Kết nối đến SQLite Database
db_file = r"E:\Ky2_2024_2025\BT_IoT_CV\db_mysql\Nhan_dien.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# 🔹 Tạo bảng Diem_danh nếu chưa có
query_create_table = """
INSERT INTO Thoigian_tiet (
    id, gio, phut
) VALUES(1, 10, 20);
"""

try:
    cursor.execute(query_create_table)
    print("✅ Thành công")

    # 🔹 Truy vấn dữ liệu
    query_select = "SELECT * FROM Thoigian_tiet"
    cursor.execute(query_select)
    rows = cursor.fetchall()

    print("📌 Dữ liệu trong bảng 'Thoigian_tiet':")
    for row in rows:
        print(row)

except sqlite3.Error as e:
    print("❌ Lỗi khi thực thi SQL:", e)

# 🔹 Đóng kết nối
conn.close()
