from channels.generic.websocket import AsyncWebsocketConsumer
import cv2

class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.accept()
        self.stream_video()

    async def disconnect(self, close_code):
        pass

    async def stream_video(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            _, buffer = cv2.imencode('.jpg', frame)
            await self.send(bytes_data=buffer.tobytes())

# Django settings file (settings.py) should include:
# ASGI_APPLICATION = 'your_project.routing.application'
