import cv2
from ultralytics import YOLO

# Load emotion model
model = YOLO("model/face_emotion_recognition.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Create fullscreen window
cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Emotion Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Run detection
    results = model(frame, conf=0.4)

    emotion_text = ""

    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy
        classes = results[0].boxes.cls

        for box, cls in zip(boxes, classes):

            x1, y1, x2, y2 = map(int, box)

            emotion = model.names[int(cls)]
            emotion_text = emotion

            # Face rectangle
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)

    # Display emotion in center
    if emotion_text:

        height, width, _ = frame.shape

        cv2.putText(frame,
                    emotion_text,
                    (width//3, height//5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0,255,0),
                    6)

    # Show frame
    cv2.imshow("Emotion Recognition", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()