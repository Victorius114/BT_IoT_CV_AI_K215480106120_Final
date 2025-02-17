import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import numpy as np

device = "cuda"
model = YOLO('test3/test1/weights/best.pt')

def xem_mat():
    file_path = filedialog.askopenfilename(
        title="Chọn một hình ảnh",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not file_path:
        print("Không có hình ảnh nào được chọn.")
        return

    image = cv2.imread(file_path)
    if image is None:
        print("Không thể đọc được ảnh.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phát hiện đối tượng bằng mô hình YOLO
    kq = model(image_rgb)

    # Lấy các bounding boxes từ kết quả phát hiện
    results = kq[0].boxes.xyxy  # Lấy tọa độ (x1, y1, x2, y2)
    if results is None or len(results) == 0:
        print("Không phát hiện người")
        return

    anh_kq = kq[0].plot()

    for box in results.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        khung = image_rgb[y1:y2, x1:x2]

    plt.figure(figsize=(10, 10))
    plt.imshow(anh_kq)
    plt.axis("off")
    plt.title("Kết quả")
    plt.show()

# Hàm nhận diện khuôn mặt từ webcam
def webcam_detection():
    cap = cv2.VideoCapture(0)  # Mở webcam mặc định (thường có chỉ số là 0)

    if not cap.isOpened():
        print("Không thể mở webcam.")
        return

    plt.ion()  # Bật chế độ hiển thị liên tục của Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    img_display = ax.imshow(np.zeros((10, 10, 3), dtype=np.uint8))  # Hình ảnh giả ban đầu
    ax.axis("off")
    plt.title("Nhận diện khuôn mặt từ webcam")

    # Cờ kiểm tra nhận diện
    is_running = True

    while is_running:
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận được khung hình.")
            break

        # Chuyển khung hình sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Phát hiện đối tượng bằng mô hình YOLO
        kq = model(frame_rgb)

        # Lấy các bounding boxes từ kết quả phát hiện
        results = kq[0].boxes.xyxy

        # Vẽ các bounding boxes lên khung hình
        for box in results.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hình chữ nhật xung quanh khuôn mặt

        # Cập nhật hình ảnh hiển thị
        img_display.set_data(frame)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Chụp lại ảnh khi có đối tượng được nhận diện
        if len(results) > 0:  # Nếu có ít nhất 1 đối tượng được nhận diện
            print("Đã nhận diện đối tượng. Chụp ảnh...")
            captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển đổi lại sang RGB để lưu hoặc hiển thị
            save_image(captured_image)  # Lưu ảnh đã chụp
            is_running = False  # Dừng nhận diện

    cap.release()
    plt.ioff()  # Tắt chế độ hiển thị liên tục
    plt.show()

# Lưu ảnh đã chụp
def save_image(captured_image):
    save_path = "captured_face_image.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))  # Lưu lại ảnh dưới định dạng BGR
    print(f"Đã lưu ảnh tại {save_path}")


# Giao diện tkinter
root = tk.Tk()
root.title("Nhận dạng khuôn mặt Demo")
root.geometry("600x400")
frame = tk.Frame(root)
frame.pack(expand=True, fill="both")


button_anh = tk.Button(frame, text="Nhận diện từ ảnh", command=xem_mat)
button_anh.pack(pady=10)

# Nút nhận diện từ webcam
button_webcam = tk.Button(frame, text="Nhận diện từ webcam", command=webcam_detection)
button_webcam.pack(pady=10)

root.mainloop()