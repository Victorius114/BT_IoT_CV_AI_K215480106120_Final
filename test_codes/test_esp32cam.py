import matplotlib.pyplot as plt
import requests
import io
from PIL import Image

ESP32_URL = "http://192.168.46.169:81/stream"

plt.ion()
fig, ax = plt.subplots()

while True:
    try:
        with requests.get(ESP32_URL, stream=True, timeout=10) as response:
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                ax.clear()
                ax.imshow(image)
                plt.pause(0.01)
            else:
                print(f"Lỗi HTTP: {response.status_code}")
                break
    except Exception as e:
        print(f"Lỗi khi nhận luồng video: {e}")
        break
