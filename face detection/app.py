from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the Flask app
app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Initialize the hand detector
detector = HandDetector(detectionCon=0.5, maxHands=2)

# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video stream generator function for Flask
def gen():
    while True:
        success, img = cap.read()
        if not success:
            break

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Use the hand detector to find hands
        hands, img = detector.findHands(img)

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            bbox1 = hand1["bbox"]
            centerPoint1 = hand1["center"]
            handType1 = hand1["type"]

            fingers1 = detector.fingersUp(hand1)

            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                centerPoint2 = hand2["center"]
                handType2 = hand2["type"]

                fingers2 = detector.fingersUp(hand2)

                # Find the distance between two hands
                length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)

        # Convert the image to JPEG format for Flask to stream it
        ret, jpeg = cv2.imencode('.jpg', img)
        if not ret:
            break

        # Yield the image as a byte stream for Flask response
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Define the route for the home page
@app.route('/')
def index():
    return render_template('C:/Users/user/Desktop/softronics/face detection/templetes/index.html')

# Define the route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
