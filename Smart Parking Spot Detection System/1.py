import cv2
import numpy as np

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Dictionary to track previous object positions
previous_positions = {}

# Load video file
video_path = r"C:\Users\akshi\Downloads\carPark.mp4"  # Raw string to handle backslashes
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Simulate depth data (in a real scenario, you would use actual depth data)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_image = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        depth_image = cv2.Canny(depth_image, 50, 150)

        # Background subtraction
        fgmask = fgbg.apply(frame)

        # Threshold the simulated depth image to detect objects within a certain range
        depth_threshold = (depth_image > 50) & (depth_image < 200)  # Simulated depth range
        thresholded_depth = np.zeros_like(depth_image)
        thresholded_depth[depth_threshold] = 255

        # Find contours in the thresholded depth image
        contours, _ = cv2.findContours(thresholded_depth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            object_center = (x + w // 2, y + h // 2)

            # Compare with previous position for behavior analysis
            if i in previous_positions:
                previous_position = previous_positions[i]
                movement = np.linalg.norm(np.array(object_center) - np.array(previous_position))

                # Check if the object is loitering (minimal movement)
                if movement < 10:
                    cv2.putText(frame, 'Loitering', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            previous_positions[i] = object_center
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the output
        cv2.imshow('Suspicious Behavior Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
