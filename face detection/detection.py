import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the hand detector
detector = HandDetector(detectionCon=0.5, maxHands=2)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    success, img = cap.read()

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
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

            length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)  

    # Display the image with hand and face detection
    cv2.imshow("Image", img)

     
    key = cv2.waitKey(1) & 0xFF  
    if key == 27:  # ESC key 
        print("Exiting the webcam feed...")
        break


cap.release()
cv2.destroyAllWindows()