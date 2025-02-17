from mtcnn import MTCNN
import cv2
import tensorflow as tf

# Kiểm tra GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Using GPU: {physical_devices[0]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected.")

# Hàm phát hiện và cắt khuôn mặt từ ảnh
def extract_face(image, required_size=(160, 160)):
    detector = MTCNN()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)
    if results:
        for result in results:
            x, y, width, height = result['box']
            x, y = max(x, 0), max(y, 0)
            face_contour = image.copy()
            cv2.rectangle(face_contour, (x, y), (x + width, y + height), (0, 255, 0), 2)
            return face_contour
    return None

# Hàm main
def main():
    # Mở webcam (0 là thiết bị webcam mặc định)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể lấy video từ webcam.")
            break

        # Phát hiện và cắt khuôn mặt trong khung hình
        face_img = extract_face(frame)

        if face_img is not None:
            # Hiển thị khuôn mặt đã cắt
            cv2.imshow("Detected Face", face_img)

        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

# Gọi hàm main
if __name__ == "__main__":
    main()
