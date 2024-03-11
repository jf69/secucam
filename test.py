import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
model.fuse()

# Open the video file
video_path = "./indoor_lighting.mp4"
# video_path = "./outdoor_dark.mp4"
# video_path = "./outdoor_day.mp4"
# video_path = "./outdoor_lightsource.mp4"
cap = cv2.VideoCapture(video_path)

# Specify width and height of our frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

classes = [0, 1, 2, 3, 5, 7, 14, 15, 16, 21, 24, 25]
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, classes=classes)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()