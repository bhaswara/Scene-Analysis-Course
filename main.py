from djitellopy import Tello
import cv2
import numpy as np 
from scipy.misc import imresize
from keras.models import load_model

#Initialize the class label
CLASS = ["door", "person"]

#initialize random color for bounding box
COLORS = np.random.uniform(0, 255, size=(len(CLASS), 3))

#This class for saving the prediction of segmentation and calculate the average
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


#This is the directory folder for saving the frames
ddir = "chk"

lanes = Lanes()
should_stop = False

class FrontEnd(object):
	def __init__(self):

		#Initialize tello
		self.tello = Tello()

		#initialize the training weight and label for detection
		self.net = cv2.dnn.readNetFromTensorflow('ssd_inception_door_person.pb', 'graph_ssd_inception_door_person.pbtxt')

		#initialize speed at starting point
		self.for_back_velocity = 0
		self.left_right_velocity = 0
		self.up_down_velocity = 0
		self.yaw_velocity = 0
		self.speed = 10

		#Initialize the control mode
		self.send_rc_control = False

		#load the training weight for segmentation
		self.model = load_model('segment_model.h5')

		#initialize the tracker
		self.tracker = cv2.TrackerKCF_create()

		#This is for checking the bounding box
		self.initBB = None


	def run(self):
		#Check if tello is not connected
		if not self.tello.connect():
			print("Tello not connected")
			return

		#Check if tello cannot stop the streaming video
		if not self.tello.streamoff():
			print("Couldn't stop video stream")
			return

		#Check if tello cannot stream the video
		if not self.tello.streamon():
			print("Couldn't start video stream")
			return

		#This is for checking the battery
		self.battery()

		#Read the video frame from tello
		frame_read = self.tello.get_frame_read()

		should_stop = False 

		#For safety zone
		szX = 100  
		szY = 55 

		#To numbering the image
		imgCount = 0


		while not should_stop:
			#set the initial timer
			timer = cv2.getTickCount()

			#update the movement of drone
			self.update()

			#check the frame
			if frame_read.stopped:
				frame_read.stop()
				break

			#convert color to rgb and read the frame
			frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
			frameRet = frame_read.frame

			############################################################################
			'''
			This code is used to do the segmentation
			'''
			#resize the frame
			framecpy = imresize(frameRet, (720, 960, 3))

			#resize the frame for segmentation, convert to array, and expand the dimension
			small_img = imresize(frameRet, (80, 160, 3))
			small_img = np.array(small_img)
			small_img = small_img[None,:,:,:]

			#predict the model
			prediction = self.model.predict(small_img)[0] * 255

			#put to the list and check last five
			lanes.recent_fit.append(prediction)
			if len(lanes.recent_fit) > 5:
				lanes.recent_fit = lanes.recent_fit[1:]

			#calculate the average
			lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

			#Generate R and B color with G stacks on it
			blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
			lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

			#resize the lane image to match the size of the frame
			lane_image = imresize(lane_drawn, (720, 960, 3))

			#overlay the main frame with the lane image
			result = cv2.addWeighted(framecpy, 1, lane_image, 1, 0)
			################################################################################

			(h, w) = result.shape[:2]

			################################################################################
			'''
			This code is used for detection
			'''
			#do the detection
			blob = cv2.dnn.blobFromImage(cv2.resize(result, (300, 300)), 0.007843, (300, 300), 127.5)
			self.net.setInput(blob)
			detect = self.net.forward()

			#This code is for detection
			for i in np.arange(0, detect.shape[2]):
				#Check the confidence
				conf = detect[0, 0, i, 2]
				
				#check if the confidence above 0.5
				if conf > 0.5:
					#take the index of detection
					idx = int(detect[0, 0, i, 1])  

					#create the bounding box
					box = detect[0, 0, i, 3:7] * np.array([w, h, w, h]) 
					(startX, startY, endX, endY) = box.astype("int")

					label = "{}: {:.2f}%".format(CLASS[idx-1], conf * 100)

					cv2.rectangle(result, (startX, startY), (endX, endY), COLORS[idx-1], 3)

					y = startY - 15 if startY - 15 > 15 else startY + 15

					cv2.putText(result, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
			####################################################################################
			
			####################################################################################
			k = cv2.waitKey(20)

			'''
			This is for controlling the movement of the drone
			'''
			#For take off the drone
			if k == ord('t'):
				print("Take Off")
				self.tello.takeoff()
				self.tello.get_battery()
				self.send_rc_control = True

			#For landing the drone
			if k == ord('l'):
				print("Landing")
				self.tello.land()
				self.send_rc_control = False

			if k == ord('e'):
				self.up_down_velocity = 50
			elif k == ord('q'):
				self.up_down_velocity = -50
			else:
				self.up_down_velocity = 0

			'''
			#This is for controlling the drone manually but we don't use it now (The flight will be automatically based on tracking)
			if k == ord('w'):
				self.for_back_velocity = 20
			elif k == ord('s'):
				self.for_back_velocity = -20
			else:
				self.for_back_velocity = 0

			if k == ord('c'):
				self.yaw_velocity = 50
			elif k == ord('z'):
				self.yaw_velocity = -50
			else:
				self.yaw_velocity = 0
		
			if k == ord('d'):
				self.left_right_velocity = 20
			elif k == ord('a'):
				self.left_right_velocity = -20
			else:
				self.left_right_velocity = 0
			'''
			#######################################################################################

			if k == 27:
				should_stop = True
				break

			#imgCount is to count the number of frames
			imgCount += 1

			########################################################################################
			'''
			This code is for creating the bounding box for tracking. If we want to initialize the tracking
			box, just press 's' and put the cursor on the object to start drawing the box.
			'''
			#Center of the dimension image window
			cWidth = int(960/2)
			cHeight = int(720/2)

			if self.initBB is not None:
				#Check if the tracker success or not
				(success, box) = self.tracker.update(result)

				if success:
					#cv2.imwrite("{}/frame{}.jpg".format(ddir,imgCount), frameRet)

					#taking the position of the box
					(x, y, w, h) = [int(v) for v in box]

					#measuring the end of box
					end_x = x + w
					end_y = y + h
					end_size = w*2

					#target coordinates
					tg_x = int((end_x + x)/2)
					tg_y = int((end_y + y)/2)

					#Calculate vector
					vTrue = np.array((cWidth, cHeight, 1026))
					vTarget = np.array((tg_x, tg_y, end_size))
					vDistance = vTrue - vTarget

					#For turning
					if vDistance[0] < -szX:
						self.yaw_velocity = 20
					elif vDistance[0] > szX:
						self.yaw_velocity = -20
					else:
						self.yaw_velocity = 0

					#For up and down
					if vDistance[1] > szY:
						self.up_down_velocity = 20
					elif vDistance[1] < -szY:
						self.up_down_velocity = -20
					else:
						self.up_down_velocity = 0

					#check distance
					F = 0
					if abs(vDistance[2]) > 1000:
						F = 20

					#For forward and backward
					if vDistance[2] > 0:
						self.for_back_velocity = 20+F
					elif vDistance[2] < 0:
						self.for_back_velocity = -20-F
					else:
						self.for_back_velocity = 0

					#Draw the bounding box
					cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

					#draw the target circle
					cv2.circle(result, (tg_x, tg_y), 10, (0,255,0),2)

					#Put Text
					yy = y - 15 if y - 15 > 15 else y + 15
					cv2.putText(result, 'Tracked', (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)


				else:
					#If not detected, set all to zero
					self.yaw_velocity = 0
					self.up_down_velocity = 0
					self.for_back_velocity = 0
					self.initBB = None
					print("No target")

			#For drawing the target
			if k == ord('s'):
				#Initialize again the target
				self.tracker = cv2.TrackerKCF_create()

				#select the point using mouse
				self.initBB = cv2.selectROI("check", result, fromCenter=False, showCrosshair=True)

				#Init again the tracker
				self.tracker.init(result, self.initBB)
			########################################################################################################

			########################################################################################################
			'''
			Show the result
			'''
			#Calculate the FPS
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

			#Add some text and FPS result
			cv2.putText(result, "FPS = %.2f" %fps, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

			#Show the result
			cv2.imshow('check', result)

			cv2.imwrite("{}/frame{}.jpg".format(ddir,imgCount), result)
			######################################################################################################

		#End the camera
		cv2.destroyAllWindows()

		#End the tello
		self.tello.end()

	def battery(self):
		#To check battery
		return self.tello.get_battery()[:2]

	def update(self):
		#To update the drone movement
		if self.send_rc_control:
			self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity)


def main():
	#Run the code
	frontend = FrontEnd()

	frontend.run()


if __name__ == '__main__':
	main()