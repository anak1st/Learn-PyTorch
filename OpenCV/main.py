import time
import cv2


print(f"OpenCV version: {cv2.__version__}")

start = time.perf_counter()
cap = cv2.VideoCapture(0)
end = time.perf_counter()
print(f"Use {(end - start):.1f} seconds to load camera")

cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
