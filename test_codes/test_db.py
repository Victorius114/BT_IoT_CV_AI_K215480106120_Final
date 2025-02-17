import os
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from fontTools.merge.util import current_time
from sympy.physics.units import microsecond
from torchvision import transforms
from scipy.spatial.distance import cosine
import pyodbc  # Thư viện kết nối SQL Server
import json
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import ttk
import mysql.connector

# Kết nối với cơ sở dữ liệu SQL Server
def connect_to_db():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=DESKTOP-VICTOR1\SQLEXPRESS;'
        'DATABASE=Nhan_dien;'
        'UID=sa;'
        'PWD=1234;'
    )
    return conn



# Lấy danh sách MSSV và thông tin từ cơ sở dữ liệu
def fetch_mssv_list(conn):
    query = "SELECT * FROM in4_sv"
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    mssv_list = [row[0] for row in rows]  # Lấy tất cả giá trị MSSV
    ten_list = [row[1] for row in rows]   # Lấy tất cả giá trị Tên
    lop_list = [row[2] for row in rows]   # Lấy tất cả giá trị Lớp
    khoa_list = [row[3] for row in rows]  # Lấy tất cả giá trị Khoa
    return mssv_list, ten_list, lop_list, khoa_list

# Hàm so sánh label với MSSV trong cơ sở dữ liệu
def compare_label_with_db(label, mssv_list, ten_list, lop_list, khoa_list):
    if label in mssv_list:
        index = mssv_list.index(label)
        info = (f"MSSV: {mssv_list[index]}"
                f"\nTên: {ten_list[index]}"
                f"\nLớp: {lop_list[index]}"
                f"\nKhoa: {khoa_list[index]}"
                f"\nThời gian: {datetime_now}")
        messagebox.showinfo("Thông tin sinh viên", info)
        data = {
            "mssv" : mssv_list[index],
            "ten" : ten_list[index],
            "lop" : lop_list[index],
            "khoa" : khoa_list[index],
            "date" : datetime_now
        }
        json_msg = json.dumps(data, ensure_ascii=False)
        messagebox.showinfo("JSON", json_msg)
        return True
    else:
        messagebox.showinfo("Thông báo", "Không tìm thấy thông tin sinh viên!")
        return False

# Khởi tạo mô hình
device = torch.device("cuda")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(device=device)

# Tải trọng số đã huấn luyện từ tệp
try:
    model.load_state_dict(torch.load('face_recognition_model.pth', map_location=device, weights_only=True))
except TypeError:
    model.load_state_dict(torch.load('face_recognition_model.pth', map_location=device))

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.ToPILImage(),  # Chuyển numpy array thành PIL image
    transforms.Resize((160, 160)),  # Resize ảnh về kích thước 160x160
    transforms.ToTensor(),  # Chuyển PIL image thành tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Chuẩn hóa
])

# Đọc dữ liệu từ thư mục và lưu embeddings
dataset_dir = r'E:\Ky2_2024_2025\BT_IoT_CV\dataset'  # Đường dẫn đến thư mục dữ liệu
dataset_embeddings = []
dataset_labels = []

for label in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, label)
    if not os.path.isdir(person_dir):
        continue
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        face = cv2.imread(image_path)
        if face is not None:
            # Phát hiện khuôn mặt và lấy embedding
            boxes, _ = mtcnn.detect(face)
            if boxes is not None:
                for box in boxes:
                    face = face[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    face_tensor = transform(face).unsqueeze(0).to(device)  # Thêm batch dimension và chuyển lên GPU
                    face_embedding = model(face_tensor).detach().cpu().numpy().flatten()  # Lấy embedding
                    dataset_embeddings.append(face_embedding)
                    dataset_labels.append(label)

dataset_embeddings = np.array(dataset_embeddings)
dataset_labels = np.array(dataset_labels)

# Kết nối tới cơ sở dữ liệu và lấy danh sách MSSV
conn = connect_to_db()
mssv_list, ten_list, lop_list, khoa_list = fetch_mssv_list(conn)

# Biến toàn cục để lưu label nhận diện được
global detected_label
detected_label = None

datetime_now = datetime.now().strftime("%d/%m/%Y "+" %H:%M:%S")
# Hàm xử lý khi nhấn nút
def on_button_click():
    global detected_label
    if detected_label:
        compare_label_with_db(detected_label, mssv_list, ten_list, lop_list, khoa_list)
        index = mssv_list.index(label)
        state = None
        time_cmpr = datetime.strptime(datetime_now, "%d/%m/%Y %H:%M:%S")
        t_dimuon = datetime_now.replace(hour=7, minute=15, second=0, microsecond=0)
        if time_cmpr >= t_dimuon:
            state = "Đi muộn"
        if time_cmpr < t_dimuon:
            state = "Đúng giờ"
        query = (f"INSERT INTO Diem_danh(mssv, [Thời gian điểm danh], [Trạng thái])"
                 f"VALUES {mssv_list[index], datetime_now, state}")
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            messagebox.showinfo("Điểm danh", "Đã điểm danh")
        except Exception as e:
            messagebox.showinfo("Điểm danh", "Không kết nối được tới cơ sở dữ liệu")
    else:
        messagebox.showinfo("Thông báo", "Không có khuôn mặt nào được nhận diện!")



def diemdanh_list():
    query = (f"SELECT in4_sv.MSSV, in4_sv.[Họ và tên], Diem_danh.[Thời gian điểm danh], Diem_danh.[Trạng thái] "
             f"FROM in4_sv "
             f"INNER JOIN Diem_danh ON in4_sv.MSSV = Diem_danh.MSSV "
             f"ORDER BY id ASC")
    try:
        df = pd.read_sql(query, conn)  # Đọc dữ liệu vào DataFrame

        # Tạo cửa sổ
        window = tk.Toplevel()
        window.title("Danh sách điểm danh")
        window.geometry("700x400")

        # Tạo bảng Treeview
        columns = list(df.columns)
        tree = ttk.Treeview(window, columns=columns, show="headings")

        # Định nghĩa tiêu đề cột
        for col in columns:
            tree.heading(col, anchor="w", text=col)
            tree.column(col, anchor="w", width=150)  # Căn giữa

        # Thêm dữ liệu vào bảng
        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

        # Thanh cuộn dọc
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        # Hiển thị bảng và thanh cuộn
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        window.mainloop()

    except Exception as e:
        tk.messagebox.showerror("Lỗi", f"Không thể tải dữ liệu: {str(e)}")


# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Nhận diện khuôn mặt")

# Tạo label để hiển thị hình ảnh từ webcam
label_video = tk.Label(root)
label_video.pack()

# Tạo nút
button = tk.Button(root, text="Điểm danh", command=on_button_click)
button.pack()
button = tk.Button(root, text="Danh sách điểm danh", command=diemdanh_list)
button.pack()


def show_webcam():
    global detected_label
    ret, frame = cap.read()
    if ret:
        # Phát hiện khuôn mặt
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            detected_label = None  # Reset lại label mỗi lần quét
            for box in boxes:
                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                # Chuyển đổi ảnh thành tensor và chuẩn hóa
                face_tensor = transform(face).unsqueeze(0).to(device)

                # Trích xuất đặc trưng khuôn mặt
                face_embedding = model(face_tensor).detach().cpu().numpy().flatten()

                # So sánh embedding của khuôn mặt nhận diện với dataset
                distances = [cosine(face_embedding, stored_embedding) for stored_embedding in dataset_embeddings]

                # Tìm label tương ứng với embedding có khoảng cách nhỏ nhất
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                label = dataset_labels[min_distance_idx]

                # **Nếu khoảng cách quá lớn, gán nhãn "Unknown"**
                threshold = 0.45  # Ngưỡng khoảng cách, điều chỉnh nếu cần
                if min_distance > threshold:
                    label = "Unknown"
                    color = (0, 0, 255)  # Đỏ nếu không nhận diện được
                else:
                    color = (0, 255, 0)  # Xanh nếu nhận diện đúng

                # Cập nhật label nhận diện được
                detected_label = label

                # Vẽ bounding box và hiển thị label
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, f"{label} ({min_distance:.2f})",
                            (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            detected_label = None  # Không có khuôn mặt -> Xóa dữ liệu nhận diện

        # Chuyển đổi hình ảnh từ OpenCV sang định dạng phù hợp với Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)

    # Lặp lại hàm sau 10ms
    label_video.after(10, show_webcam)

# Bắt đầu hiển thị webcam
show_webcam()

# Chạy vòng lặp chính của Tkinter
root.mainloop()

# Giải phóng webcam và đóng cửa sổ OpenCV
cap.release()
cv2.destroyAllWindows()