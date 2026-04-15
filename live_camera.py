import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

print("Starting live camera... Press 'q' on your keyboard to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)

    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        if not isinstance(results, list):
            results = [results]

        for face in results:
            if face['face_confidence'] > 0:
                x, y = face['region']['x'], face['region']['y']
                w, h = face['region']['w'], face['region']['h']

                dominant_emotion = face['dominant_emotion']
                confidence = face['emotion'][dominant_emotion]

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                label = f"{dominant_emotion.capitalize()} ({confidence:.1f}%)"
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    except Exception:
        pass

    cv2.imshow(
        'Live Emotion Detector (Press "q" to quit)',
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()