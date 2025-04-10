import cv2
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO('runs/detect/train/weights/best.pt')  # Use your trained model

# Class names (Ensure these match your model's classes)
classnames = ["Fall", "Sitting", "Falling", "Standing"]

# Define input source:
# - Set to a video file path (e.g., "fall_vdo_testing.mp4") for video detection
# - Set to an image file path (e.g., "test_image.jpg") for image detection
# - Set to None for webcam mode
input_source = None  # 'J fall.MOV' or ''fall_vdo_testing.mp4    Change this as needed 
# Function to process frames (for video, webcam, and images)
def process_frame(frame):
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            confidence = box.conf[0].item()  # Get confidence score
            class_id = int(box.cls[0].item())  # Get class index
            class_name = classnames[class_id]  # Get class name

            # Set color (Red for Fall, Green for others)
            color = (0, 255, 0) if class_name != "Fall" else (0, 0, 255)
            
            # Draw bounding box (Thicker border)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)  # Thicker box
            
            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2  # Bigger text
            font_thickness = 3  # Thicker text

            # Add a background rectangle for text
            label = f"{class_name} ({confidence:.2f})"
            (w, h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)  # Background for text
            cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

            # Alert for Fall detection
            if class_name == "Fall":
                print("⚠️ Fall detected!")

    return frame

# Check the input type
if input_source:  # Video or Image Mode
    if input_source.lower().endswith(('.mp4', '.avi', '.mov')):  # Video mode
        cap = cv2.VideoCapture(input_source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video ends

            processed_frame = process_frame(frame)
            cv2.imshow('Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Press 'q' to exit

        cap.release()

    elif input_source.lower().endswith(('.jpg', '.png', '.jpeg')):  # Image mode
        frame = cv2.imread(input_source)
        processed_frame = process_frame(frame)
        cv2.imshow('Detection', processed_frame)
        cv2.waitKey(0)  # Wait for user to close window

else:  # Webcam Mode
    cap = cv2.VideoCapture(0)  # Change to 1 if using an external webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow('Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit

    cap.release()

cv2.destroyAllWindows()

