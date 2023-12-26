# Load modules
import os
import cv2
import yolov5
import Jetson.GPIO as GPIO

# Load YOLOv5 Model
model = yolov5.load("/home/ics/Workspace/WhiteCane-AI/yolov5/runs/train/result/weights/best.pt")

# Define button pin number
PIN = 15

# Set GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN, GPIO.IN)

# object class and class name mapping
class_mapping = {
		0.0: "pepsi can",
		1.0: "pepsi pet",
		2.0: "cocacola can",
		3.0: "cocacola pet"
}

# object detection function
def object_detect():
	# image capture
	camera = cv2.VideoCapture(0)
	ret, frame = camera.read()
	cv2.imwrite("captured_image.jpg", frame)
	camera.release()

	# image load
	image_path = "captured_image.jpg"
	results = model(image_path)

	# print result
	max_size = 0
	max_box = None
	for result in results.pred[0]:
		x1, y1, x2, y2 = result[:4]
		box_width = x2 - x1
		box_height = y2 - y1
		box_size = box_width * box_height
		if box_size > max_size:
			max_size = box_size
			max_box = result
	
	if max_box is not None:
		class_index = int(max_box[5])
		class_name = class_mapping[class_index]
		print(f"The size of the largest box : {max_size}")
		print(f"Detected objetc name: {class_name}")
		os.system(f"echo {class_name} | festival --tts")
	else:
		result = "Not detected"
		print(result)
		os.system(f"echo {result} | festival --tts")

print("Ready for object detection")

while True:
	if GPIO.input(PIN) == 0: # Push button
		object_detect()
	else:
		pass
