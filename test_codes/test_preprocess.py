import cv2
import os
import numpy as np


def preprocess_image(image_path, output_dir, image_name):
    # Đọc ảnh
    img = cv2.imread(image_path)

    # Nếu ảnh không thể đọc được, trả về None
    if img is None:
        return None

    # Chuyển đổi ảnh sang màu xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sử dụng OpenCV để phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Cắt khuôn mặt ra khỏi ảnh
        face = img[y:y + h, x:x + w]

        # Tạo tên tệp mới và lưu ảnh khuôn mặt
        output_path = os.path.join(output_dir, f"{image_name}_face.jpg")
        cv2.imwrite(output_path, face)
        return output_path  # Trả về đường dẫn ảnh đã lưu

    return None


def process_directory(input_dir, output_dir):
    # Kiểm tra nếu thư mục đầu ra không tồn tại, tạo mới
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lặp qua tất cả các tệp trong thư mục đầu vào
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # Kiểm tra nếu là tệp ảnh (ở đây giả sử ảnh có đuôi .jpg hoặc .png)
        if filename.lower().endswith(('.jpg', '.png')):
            print(f"Processing {filename}...")

            # Xử lý ảnh
            output_path = preprocess_image(file_path, output_dir, os.path.splitext(filename)[0])

            if output_path is not None:
                print(f"Face saved to {output_path}")
            else:
                print(f"No face found in {filename}")


# Đường dẫn thư mục chứa các ảnh gốc và thư mục lưu ảnh khuôn mặt
input_dir = "/dataset/K215480106120"
output_dir = "/dataset/K215480106120/LaDucThang_faces"
process_directory(input_dir, output_dir)
