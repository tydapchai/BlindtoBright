import cv2
import requests
import numpy as np

url = "http://192.168.0.183:81/stream"

response = requests.get(
    url,
    stream=True,
    timeout=(3, 5),
)

buffer = bytearray()

for chunk in response.iter_content(4096):
    buffer.extend(chunk)

    start = buffer.find(b"\xff\xd8")
    end = buffer.find(b"\xff\xd9", start + 2)

    if start == -1 or end == -1:
        continue

    jpg = bytes(buffer[start:end + 2])
    del buffer[:end + 2]

    frame = cv2.imdecode(
        np.frombuffer(jpg, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        continue

    cv2.imshow("Raw ESP32 stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break