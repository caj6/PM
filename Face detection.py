import cv2
import os

# Get the path to the Haar Cascade files
opencv_data_path = cv2.__path__[0]
face_cascade_path = os.path.join(opencv_data_path, 'data', 'haarcascade_frontalface_default.xml')

# Load the Haar Cascade files
face_cascade = cv2.CascadeClassifier(face_cascade_path)


# Check if the cascade files were loaded successfully
if face_cascade.empty():
    print(f"Error: Could not load Haar Cascade file: {face_cascade_path}")
    exit()
    
# Check available cameras
for i in range(0, 2):  # Check the first 2 indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available.")
        cap.release()
    else:
        print(f"No camera found at index {i}.")
        
# Initialize the webcam
cam = cv2.VideoCapture(1)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a named window
window_name = 'Face Detection'
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while True:
    # Read a frame from the webcam
    ret, img = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow(window_name, img)

    # Exit if 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()