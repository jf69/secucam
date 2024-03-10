import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

import supervision as sv

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from email_settings import password, from_email, to_email

# Create server. Use gmail
server = smtplib.SMTP('smtp.gmail.com: 587')

# Start server
server.starttls()

# Login credentials for sending the mail
server.login(from_email, password)

def send_email(to_email, from_email, object_detected=1):
	message = MIMEMultipart()
	message['From'] = from_email
	message['To'] = to_email
	message['Subject'] = "Security Alert"

	# add in the message body
	message_body = f'ALERT - {object_detected} object(s) detected.'
	message.attach(MIMEText(message_body, 'plain'))

	# Send the email
	server.sendmail(from_email, to_email, message.as_string())

class ObjectDetection:
	def __init__(self, capture_index): # here the capture index will be webcam
		self.capture_index = capture_index
		
		self.email_sent = False
		
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		print("Using Device: ", self.device)
		
		self.model = self.load_model()
		
		self.annotator = None

		self.start_time = 0
		self.end_time = 0

		self.CLASS_NAMES_DICT = self.model.model.names

		# class for drawing bounding boxes on an image using detections provided
		self.annotator = sv.BoxAnnotator(
			color=sv.Color.WHITE,
			thickness=3,
			text_thickness=3,
			text_scale=1.5
		)

	def load_model(self):
		model = YOLO("yolov8s.pt") # load a pretrained YOLOv8n model
		model.fuse()
		return model

	def predict(self, frame):
		results = self.model.track(frame, conf=0.5)
		return results
	
	def display_fps(self, frame):
		self.end_time = time()
		
		fps = 1 / np.round(self.end_time - self.start_time, 2)
		
		text = f'FPS: {int(fps)}'
		loc = (20,70)
		font = cv2.FONT_HERSHEY_SIMPLEX
		scale = 1.5
		color = (0,255,0)
		thickness = 2
		
		cv2.putText(frame, text, loc, font, scale, color, thickness)

	def plot_bboxes(self, results, frame):
		xyxys = []
		confidences = []
		class_ids = []

		# self.annotator = Annotator(frame, 3, results[0].names)
		
		boxes = results[0].boxes.xyxy.cpu()
		clss = results[0].boxes.cls.cpu().tolist()
		names = results[0].names

		for box, cls in zip(boxes, clss):
			class_ids.append(cls)
			# self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
		
		# return frame, class_ids

		# # Extract detections for person class
		# for result in results[0]:
		# 	class_id = result.boxes.cls.cpu().numpy().astype(int)

		# 	if class_id == 0: # if class is person
		# 		xyxys.append(result.boxes.xyxy.cpu().numpy())
		# 		confidences.append(result.boxes.conf.cpu().numpy())
		# 		class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

		# Set up detections for visualization from supervision.
		# This will simply format, display and annotate our frame
		# detections = sv.Detections.from_ultralytics(results[0])

		# frame = self.annotator.annotate(scene=frame, detections=detections)
		detections = sv.Detections(
                    xyxy=boxes.numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )

		# Format custom labels
		# self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for _, confidence, class_id, tracker_id in detections]

		# Annotate and display frame
		frame = self.annotator.annotate(scene=frame, detections=detections)#, labels=self.labels)

		return frame, class_ids

	def __call__(self):
		# Open up a video capture
		cap = cv2.VideoCapture(self.capture_index)
		assert cap.isOpened()

		# Specify width and height of our frame
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

		frame_count = 0

		while True: # as long as we don't hit q on the keyboard to terminate program
			self.start_time = time()

			# load in a frame from camera
			ret, frame = cap.read()
			assert ret

			# get the results on the frame
			results = self.predict(frame)
			
			# Plot the bounding boes on the frame and return the class ids along with the frame
			frame, class_ids = self.plot_bboxes(results, frame)

			# Logic controlling our emails
			if len(class_ids) > 0:
				if not self.email_sent: # Only send email if it has not been sent for the current detection
					send_email(to_email, from_email, len(class_ids))
					self.email_sent = True # Set the flag to True adter sending the email
			else:
				self.email_sent = False # Reset the flag when no person is detected

			self.display_fps(frame)

			# end_time = time()

			# fps = 1/np.round(end_time - start_time, 2)

			# text = f'FPS: {int(fps)}'
			# loc = (20,70)
			# font = cv2.FONT_HERSHEY_SIMPLEX
			# scale = 1.5
			# color = (0,255,0)
			# thickness = 2

			# cv2.putText(frame, text, loc, font, scale, color, thickness)

			cv2.imshow('YOLOv8 Detection', frame)

			frame_count += 1

			if cv2.waitKey(5) & 0xFF == 27:
				break
		
		# Release our webcam, destroy windows opened by opencv, and quit the server
		cap.release()
		cv2.destroyAllWindows()
		server.quit()

# Create instance of our class and specifiy webcam index
detector = ObjectDetection(capture_index=0)
# call the detector function
detector()
