import cv2
import numpy as np

def preprocess_image(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)

    # Chuyển đổi ảnh sang màu xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sử dụng OpenCV để phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Cắt khuôn mặt ra khỏi ảnh
        face = img[y:y + h, x:x + w]
        return face  # Trả về ảnh khuôn mặt
    return None

lmao = preprocess_image("E:\\Ky2_2024_2025\\BT_IoT_CV\\dataset\\LaDucThang_K215480106120\\1.jpg")
cv2.imshow("lmaop", lmao)
cv2.waitKey(0)