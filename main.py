import cv2

# Path to the pre-trained Haar Cascade classifier for face detection
harcascade = "model/haarcascade_frontalface_default.xml"

# Open the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Set the resolution of the video capture
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    # Read a frame from the video capture
    success, img = cap.read()

    # Check if the frame was successfully read
    if not success:
        print("Failed to read frame")
        break

    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(harcascade)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with detected faces
    cv2.imshow("Face Detection", img)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
