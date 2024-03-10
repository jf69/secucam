import torch
from ultralytics import YOLO
import cv2
import numpy as np

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email_settings import password, from_email, to_email

from time import time
from datetime import datetime

# Create server. Use gmail
server = smtplib.SMTP('smtp.gmail.com: 587')

# Start server
server.starttls()

# Login credentials for sending the mail
server.login(from_email, password)

def send_email(to_email, from_email, time, class_name, track_id, confidence, xyxy):
	message = MIMEMultipart()
	message['From'] = from_email
	message['To'] = to_email
	message['Subject'] = "Security Alert Update"

	# add in the message body
	message_body = f'{class_name} with tracking id {track_id} and confidence {confidence:{3}.{2}} was detected at coordinates {xyxy} on {time}'

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
		
		self.max_track_id = 0

		self.start_time = 0
		self.end_time = 0

		self.CLASS_NAMES_DICT = self.model.model.names

	def load_model(self):
		model = YOLO("yolov8n.pt") # load a pretrained YOLOv8n model
		model.fuse()
		return model

	def track(self, frame):
		# Only detect person, bicycle, car, moto, bus, truck, bird, cat, dog, bear, backpack, umbrella, respectively.
		classes = [0, 1, 2, 3, 5, 7, 14, 15, 16, 21, 24, 25]
		results = self.model.track(frame, persist=True,  tracker="bytetrack.yaml", conf=0.6, classes=classes)
		return results
	
	def display_fps(self, frame):
		self.end_time = time()
		
		fps = 1 / np.round(self.end_time - self.start_time, 2)
		
		text = f'FPS: {int(fps)}'
		loc = (10,30)
		font = cv2.FONT_HERSHEY_SIMPLEX
		scale = 1
		color = (0,255,0)
		thickness = 1
		
		cv2.putText(frame, text, loc, font, scale, color, thickness)

	def plot_bboxes(self, results, frame):
		xyxys = []
		confidences = []
		classes = []
		track_ids = []
	
		bboxes = results[0].boxes

		xyxys = bboxes.xyxy.cpu().numpy().astype(int) # coords of each detected bounding box in an ndarray
		class_ids = bboxes.cls.cpu().numpy().astype(int) # class ID for each detected bounding box in an ndarray
		confidences = bboxes.conf.cpu().numpy() # confidence for each detected bounding box in an ndarray
		track_ids = bboxes.id.int().cpu().tolist() if bboxes.id != None else [] # tracking ID for each detected bounding box in an ndarray
		
		frame = results[0].plot(conf=False, kpt_line=False, masks=False, probs=False)

		print(xyxys)
		print(class_ids)
		print(confidences)
		print(track_ids)
		
		return frame, xyxys, class_ids, confidences, track_ids

	def __call__(self):
		# Open up a video capture
		cap = cv2.VideoCapture(self.capture_index)
		assert cap.isOpened()

		# Specify width and height of our frame
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

		frame_count = 0

		while cap.isOpened():
			self.start_time = time()

			# load in a frame from camera
			ret, frame = cap.read()
			assert ret

			# get the results on the frame
			results = self.track(frame)
			
			# Plot the bounding boes on the frame and return the class ids along with the frame
			frame, xyxys, class_ids, confidences, track_ids = self.plot_bboxes(results, frame)

			# Logic controlling our emails
			if len(track_ids) > 0:
				last_track_id = track_ids[-1]
				if  last_track_id > self.max_track_id:
					detection_time = datetime.now().strftime('%d%b%Y-%HH%MM%SS')

					send_email(
						to_email,
						from_email,
						time=detection_time,
						class_name=self.CLASS_NAMES_DICT[class_ids[-1]],
						track_id=last_track_id,
						confidence=confidences[-1],
						xyxy=xyxys[-1]
					)
					self.max_track_id = last_track_id

			self.display_fps(frame)

			cv2.imshow('YOLOv8 Detection', frame)

			frame_count += 1

			# Break loop if q is pressed
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
