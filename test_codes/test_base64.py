import base64
import cv2
import numpy as np


def decode_base64_to_image(file_path):
    """Đọc chuỗi Base64 từ file .txt, giải mã và hiển thị ảnh bằng OpenCV."""
    try:
        with open(file_path, "r") as file:
            base64_string = file.read().strip()

        image_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is not None:
            cv2.imshow("Decoded Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Lỗi: Không thể giải mã ảnh")
    except Exception as e:
        print(f"Lỗi: {e}")


# Gọi hàm với đường dẫn file chứa chuỗi Base64
decode_base64_to_image("E:\Ky2_2024_2025\BT_IoT_CV\encoded_image.txt")
