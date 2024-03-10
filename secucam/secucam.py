import torch
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email_settings import password, from_email, to_email

import os
import sys
import django

from time import time
from datetime import datetime as dt, timedelta as td

# Add the package list to the python package search directories.
sys.path.append('.')
# Tell it where the settings file for our project is.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'secucam.settings')
# Initialize the django system.
django.setup()

from secucam_app.models import *

# Create server. Use gmail
server = smtplib.SMTP('smtp.gmail.com: 587')

# Start server
server.starttls()

# Login credentials for sending the mail
server.login(from_email, password)

# Function to send email alert upon detection
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

# Class for object detection
class ObjectDetection:
	def __init__(self, capture_index):
		# The capture index will is the input video source (webcam, mp4, etc.)
		self.capture_index = capture_index
		
		# Check if device is using Nvidia cuda or cpu.
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		print("Using Device: ", self.device)
		
		# Load the model
		self.model = self.load_model()
		
		# Counter for the highest track_id
		self.max_track_id = 0

		# Used to calculate fps
		self.start_time = 0
		self.end_time = 0

		# Dict of all the detectable class names
		self.CLASS_NAMES_DICT = self.model.model.names

		# Used to keep track of saved footage
		self.recording = False
		self.recording_start_time = 0

		# List to store the frames during recording
		self.saved_frames = []

	# Function to load the model
	def load_model(self):
		# load a pretrained YOLOv8n model
		model = YOLO("./yolov8_models/yolov8n.pt")
		model.fuse()
		return model

	# Function to start tracking using the loaded model
	def track(self, frame):
		# Only detect person, bicycle, car, moto, bus, truck, bird, cat, dog, bear, backpack, umbrella, respectively.
		classes = [0, 1, 2, 3, 5, 7, 14, 15, 16, 21, 24, 25]
		results = self.model.track(frame, persist=True,  tracker="bytetrack.yaml", conf=0.5, classes=classes)
		return results
	
	# Function to calculate and display fps
	def display_fps(self, frame):
		# Get time
		self.end_time = time()
		
		# Calculate fps
		fps = 1 / np.round(self.end_time - self.start_time, 2)
		
		text = f'FPS: {int(fps)}'
		loc = (10,30)
		font = cv2.FONT_HERSHEY_SIMPLEX
		scale = 1
		color = (0,255,0)
		thickness = 1
		# Use cv2 to draw fps text on frame
		cv2.putText(frame, text, loc, font, scale, color, thickness)

	# Function to etract results and draw bounding boxes
	def plot_bboxes(self, results, frame):
		# results[0] is an object that contains all detections in a frame.
		# From this object, we are interested in the bounding boxes
		bboxes = results[0].boxes

		# Extract the array containing the coords of each detected bounding box
		xyxys = bboxes.xyxy.cpu().numpy().astype(int)
		# Extract the array containing the class ID of each detected bounding box
		class_ids = bboxes.cls.cpu().numpy().astype(int)
		# Extract the array containing the confidence of each detected bounding box
		confidences = bboxes.conf.cpu().numpy()
		# Extract the array containing the tracking ID of each detected bounding box, if that array is not empty.
		track_ids = bboxes.id.int().cpu().tolist() if bboxes.id != None else []
		
		# Draw the detected bounding boxes, along with classes and tracking IDs, on the frame
		frame = results[0].plot(conf=False, kpt_line=False, masks=False, probs=False)

		# Print the arrays to console for debugging
		print(xyxys)
		print(class_ids)
		print(confidences)
		print(track_ids)
		
		# Return the extracted arrays
		return frame, xyxys, class_ids, confidences, track_ids

	def __call__(self):
		# Open up a video capture
		cap = cv2.VideoCapture(self.capture_index)
		assert cap.isOpened()

		# Specify width and height of our frame
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

		# While video is on
		while cap.isOpened():
			self.start_time = time()

			# load in a frame from camera
			ret, frame = cap.read()
			assert ret

			# get the results on the frame
			results = self.track(frame)
			
			# Plot the bounding boxes on the frame and return the box info along with the frame
			frame, xyxys, class_ids, confidences, track_ids = self.plot_bboxes(results, frame)

			# Logic for sending an email and setting the recording flag when a new detection occurs
			# If there is a detection
			if len(track_ids) > 0:
				# Get the last tracking ID (i.e. largest) from the track_ids array.
				last_track_id = track_ids[-1]

				# If that ID is larger than the previous largest tracking ID, then a new detection has occurred
				if  last_track_id > self.max_track_id:

					# Get the current date and time in string form
					detection_time = dt.now().strftime('%d-%m-%Y-%Hh%Mm%Ss')

					# Send an email to the user with info on the new detection
					send_email(
						to_email,
						from_email,
						time=detection_time,
						class_name=self.CLASS_NAMES_DICT[class_ids[-1]],
						track_id=last_track_id,
						confidence=confidences[-1],
						xyxy=xyxys[-1]
					)

					# Update the previous largest tracking ID to the latest one.
					self.max_track_id = last_track_id
					
					# Set the flag to start recording
					self.recording = True
					# Set the time at which the recording starts
					self.recording_start_time = dt.now()
					print('STARTED RECORDING')
			
			# Check if we are recording
			if self.recording:
				# As long as the recording time has not exceeded 20 seconds and that there are still detected objects in the frame
				if (dt.now() < self.recording_start_time + td(0,20)) and (len(track_ids) > 0):
					# Add the frame to the saved_frames array
					self.saved_frames.append(frame)
					print("RECORDING")
				# Otherwise, when the 20 seconds after recording_start_time have elapsed or when all detected objects are no longer in the frame
				else:
					# Move to the footage directory
					os.chdir("./secucam_app/footage")
					# print(os.getcwd())

					# Create a VideoWriter object with cv2
					# Set the name of the video to the recording_start_time
					video_name = f"{self.recording_start_time.strftime('%d-%m-%Y-%Hh%Mm%Ss')}.avi"
					# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
					fps = 30
					video = cv2.VideoWriter(video_name, -1, fps, (640, 480))

					# Iterate over the frames in the saved_frames array and combine to convert them into video
					for frame in self.saved_frames:
						video.write(frame)

					# Create a new row in the Footage database table and store the video_name
					row = Footage.objects.create(footage_name=video_name)
					# Save the new entry
					row.save()

					# Set the recording flag to false
					self.recording = False
					# Empty the saved_frames array.
					self.saved_frames = []
					print('STOPPED RECORDING')

					# Return to the original directory secucam
					os.chdir("../..")
					# print(os.getcwd())

			# Call the function to display the fps
			self.display_fps(frame)

			# Display the live feed with cv2
			cv2.imshow('YOLOv8 Detection', frame)

			# Break loop if escape is pressed
			if cv2.waitKey(5) & 0xFF == 27:
				break
		
		# Release the webcam, destroy windows opened by opencv, and quit the server
		cap.release()
		cv2.destroyAllWindows()
		server.quit()

# Create instance of our class and specifiy webcam index
detector = ObjectDetection(capture_index=0)
# call the detector function
detector()