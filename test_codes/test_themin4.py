import pyodbc
import sqlite3

# 🔹 Kết nối SQL Server
sql_server_conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=DESKTOP-VICTOR1\\SQLEXPRESS;"
    "DATABASE=Nhan_dien;"
    "UID=sa;"
    "PWD=1234;"
)
sql_server_cursor = sql_server_conn.cursor()

# 🔹 Kết nối SQLite
sqlite_conn = sqlite3.connect(r"E:\Ky2_2024_2025\BT_IoT_CV\db_mysql\Nhan_dien.db")
sqlite_cursor = sqlite_conn.cursor()

# 🔹 Lấy danh sách bảng trong SQL Server
sql_server_cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
tables = [row[0] for row in sql_server_cursor.fetchall()]

for table in tables:
    print(f"🔄 Đang xử lý bảng: {table}")

    # 🔹 Lấy cấu trúc bảng từ SQL Server
    sql_server_cursor.execute(
        f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}'")
    columns = sql_server_cursor.fetchall()

    column_definitions = []
    for col_name, data_type in columns:
        if data_type in ("int", "bigint", "smallint", "tinyint"):
            col_type = "INTEGER"
        elif data_type in ("nvarchar", "varchar", "nchar", "char", "text"):
            col_type = "TEXT"
        elif data_type in ("datetime", "date", "smalldatetime"):
            col_type = "TEXT"  # SQLite lưu datetime dưới dạng chuỗi
        elif data_type in ("float", "decimal", "numeric", "real"):
            col_type = "REAL"
        else:
            col_type = "BLOB"  # Dữ liệu nhị phân

        column_definitions.append(f'"{col_name}" {col_type}')

    # 🔹 Xóa bảng nếu đã tồn tại và tạo lại trong SQLite
    sqlite_cursor.execute(f"DROP TABLE IF EXISTS {table}")
    create_table_query = f"CREATE TABLE {table} ({', '.join(column_definitions)})"
    sqlite_cursor.execute(create_table_query)

    # 🔹 Lấy dữ liệu từ SQL Server
    sql_server_cursor.execute(f"SELECT * FROM {table}")
    rows = sql_server_cursor.fetchall()

    # 🔹 Chèn dữ liệu vào SQLite
    placeholders = ", ".join(["?" for _ in columns])
    insert_query = f"INSERT INTO {table} VALUES ({placeholders})"
    sqlite_cursor.executemany(insert_query, rows)

    print(f"✅ Đã chuyển bảng {table} thành công!")

# 🔹 Lưu thay đổi và đóng kết nối
sqlite_conn.commit()
sqlite_conn.close()
sql_server_conn.close()
print("🎉 Hoàn thành chuyển đổi dữ liệu!")
