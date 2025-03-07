import cv2
import numpy as np
import pyrealsense2 as rs


# Paths to the Caffe model files
model_path = r"C:\Users\junio\Downloads\opencv_face_detector_uint8.pb"
config_path = r"C:\Users\junio\Downloads\opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

# Create a named window
window_name = "Face Detection"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Get the frame dimensions
        (h, w) = color_image.shape[:2]

        # Prepare the input blob for the DNN
        blob = cv2.dnn.blobFromImage(
            color_image, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False
        )
        net.setInput(blob)

        # Perform face detection
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.7:
                # Compute the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box around the face
                cv2.rectangle(
                    color_image, (startX, startY), (endX, endY), (255, 0, 0), 2
                )

        # Display the frame
        cv2.imshow(window_name, color_image)

        # Exit if 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Stop the pipeline and close all OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
