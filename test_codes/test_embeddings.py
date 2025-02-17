import cv2
import tensorflow as tf
import numpy as np
import os
from scipy.spatial.distance import cosine
import test_preprocess as pr  # Đảm bảo module này hoạt động đúng

# Tải mô hình FaceNet đã huấn luyện
model_path = r'E:\Ky2_2024_2025\BT_IoT_CV\facenet_keras.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy tệp mô hình tại: {model_path}")
model = tf.keras.models.load_model(model_path)  # Sửa ở đây

def get_embedding(face_image):
    """
    Chuyển đổi ảnh khuôn mặt thành embedding bằng mô hình FaceNet.

    Args:
        face_image (numpy.ndarray): Ảnh khuôn mặt đã cắt.

    Returns:
        numpy.ndarray: Embedding của khuôn mặt.
    """
    face_image = cv2.resize(face_image, (160, 160))  # Resize ảnh khuôn mặt
    face_image = np.expand_dims(face_image, axis=0)
    face_image = face_image.astype('float32') / 255.0  # Chuẩn hóa giá trị pixel
    embedding = model.predict(face_image)
    return embedding.flatten()

def get_embeddings_for_dataset(dataset_dir):
    """
    Lấy embedding và nhãn cho tất cả các ảnh trong dataset.

    Args:
        dataset_dir (str): Đường dẫn tới thư mục dataset.

    Returns:
        tuple: (numpy.ndarray embeddings, numpy.ndarray labels)
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset tại: {dataset_dir}")

    embeddings = []
    labels = []
    for label in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            face = pr.preprocess_image(image_path)  # Xử lý ảnh
            if face is not None:
                try:
                    embedding = get_embedding(face)
                    embeddings.append(embedding)
                    labels.append(label)
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
    return np.array(embeddings), np.array(labels)

# Ví dụ sử dụng hàm:
dataset_dir = r"/dataset/K215480106120/LaDucThang_faces"  # Đường dẫn tới dataset
try:
    embeddings, labels = get_embeddings_for_dataset(dataset_dir)
    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:", labels.shape)
except Exception as e:
    print(f"Lỗi trong quá trình xử lý dataset: {e}")
