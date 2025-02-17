import os
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import cv2
from torchvision import transforms
from scipy.spatial.distance import cosine

# Kiểm tra và chọn thiết bị GPU/CPU
device = torch.device("cuda")

# Khởi tạo mô hình
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Dùng pretrained model làm nền tảng
mtcnn = MTCNN(device=device)

# Tải trọng số đã huấn luyện từ tệp
try:
    model.load_state_dict(torch.load('face_recognition_model.pth', map_location=device, weights_only=True))  # Sử dụng weights_only=True nếu được hỗ trợ
except TypeError:
    model.load_state_dict(torch.load('face_recognition_model.pth', map_location=device))  # Fallback nếu weights_only không được hỗ trợ

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

# Tiến hành nhận dạng khuôn mặt qua webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện khuôn mặt
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            # Chuyển đổi ảnh thành tensor và chuẩn hóa
            face_tensor = transform(face).unsqueeze(0).to(device)  # Thêm batch dimension và chuyển lên GPU

            # Trích xuất đặc trưng khuôn mặt
            face_embedding = model(face_tensor).detach().cpu().numpy().flatten()

            # So sánh embedding của khuôn mặt nhận diện với các embedding trong dataset
            distances = []
            for stored_embedding in dataset_embeddings:
                dist = cosine(face_embedding, stored_embedding)
                distances.append(dist)

            # Tìm label tương ứng với embedding có khoảng cách nhỏ nhất
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            label = dataset_labels[min_distance_idx]

            # Đo độ dài của văn bản
            (text_width, text_height), _ = cv2.getTextSize(f"Match: {label}", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            # Tính toán tọa độ bắt đầu của văn bản sao cho căn giữa
            x = int(box[0]) - text_width // 3
            y = int(box[1]) - 10

            # Kiểm tra nếu khoảng cách nhỏ hơn ngưỡng cho phép (ví dụ 0.6)
            if min_distance < 0.6:


                # Vẽ văn bản với căn giữa
                cv2.putText(frame, f"{label}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:
                cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Vẽ bounding box trên ảnh
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
