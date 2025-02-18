import os
import numpy as np
import torch
import cv2
import pyodbc
import json
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
from torchvision import transforms
import pandas as pd

app = Flask(__name__)


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
    ten_list = [row[1] for row in rows]  # Lấy tất cả giá trị Tên
    lop_list = [row[2] for row in rows]  # Lấy tất cả giá trị Lớp
    khoa_list = [row[3] for row in rows]  # Lấy tất cả giá trị Khoa
    return mssv_list, ten_list, lop_list, khoa_list


# Khởi tạo mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(device=device)

# Tải trọng số đã huấn luyện từ tệp
model.load_state_dict(torch.load('face_recognition_model.pth', map_location=device, weights_only=True))

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
            boxes, _ = mtcnn.detect(face)
            if boxes is not None:
                for box in boxes:
                    face = face[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    face_tensor = transform(face).unsqueeze(0).to(device)
                    face_embedding = model(face_tensor).detach().cpu().numpy().flatten()
                    dataset_embeddings.append(face_embedding)
                    dataset_labels.append(label)

dataset_embeddings = np.array(dataset_embeddings)
dataset_labels = np.array(dataset_labels)

# Kết nối tới cơ sở dữ liệu và lấy danh sách MSSV
conn = connect_to_db()
mssv_list, ten_list, lop_list, khoa_list = fetch_mssv_list(conn)

# Biến toàn cục để lưu label nhận diện được
detected_label = None


# Hàm nhận diện khuôn mặt từ webcam
def detect_face():
    global detected_label
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            detected_label = None  # Reset lại label mỗi lần quét
            for box in boxes:
                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                face_tensor = transform(face).unsqueeze(0).to(device)
                face_embedding = model(face_tensor).detach().cpu().numpy().flatten()

                distances = [cosine(face_embedding, stored_embedding) for stored_embedding in dataset_embeddings]
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                label = dataset_labels[min_distance_idx]
                threshold = 0.5  # Ngưỡng khoảng cách, điều chỉnh nếu cần
                if min_distance > threshold:
                    label = "Unknown"
                    color = (0, 0, 255)  # Đỏ nếu không nhận diện được
                else:
                    color = (0, 255, 0)  # Xanh nếu nhận diện đúng
                detected_label = label

                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index_gv.html')


@app.route('/video_feed')
def video_feed():
    return Response(detect_face(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/diemdanh', methods=['GET', 'POST'])
def diemdanh():
    global detected_label
    if detected_label:
        # Fetch data from database and save attendance
        index = mssv_list.index(detected_label)
        state = "Đúng giờ" if datetime.now().hour < 7 else "Đi muộn"
        query = f"INSERT INTO Diem_danh(mssv, [Thời gian điểm danh], [Trạng thái]) VALUES (?, ?, ?)"
        datetime_now = datetime.now().strftime("%d/%m/%Y " + " %H:%M:%S")
        time_cmpr = datetime.strptime(datetime_now, "%d/%m/%Y %H:%M:%S")
        cursor = conn.cursor()
        cursor.execute(query, (mssv_list[index], time_cmpr, state))
        conn.commit()
        json_msg = jsonify(
            {'mssv' : f'{mssv_list[index]}',
             'time' : time_cmpr,
             'status': 'success',
             'message': 'Đã điểm danh'}
        )
        print(json_msg.get_json())  # In JSON ra console
        return json_msg
    return jsonify({'status': 'fail', 'message': 'Không nhận diện được khuôn mặt'})


@app.route('/diemdanh_list')
def diemdanh_list():
    query = (f"SELECT in4_sv.MSSV, in4_sv.[Họ và tên], Diem_danh.[Thời gian điểm danh], Diem_danh.[Trạng thái] "
             f"FROM in4_sv "
             f"INNER JOIN Diem_danh ON in4_sv.MSSV = Diem_danh.MSSV "
             f"ORDER BY id ASC")
    df = pd.read_sql(query, conn)  # Đọc dữ liệu vào DataFrame
    df.reset_index(drop=True, inplace=True)

    # Chuyển bảng thành HTML và loại bỏ khoảng trắng
    table_html = df.to_html(classes='table table-bordered', index=False).replace('\n', '')

    return render_template('diemdanh_list.html', tables=table_html)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
