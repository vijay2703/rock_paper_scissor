import os
import cv2
import sys
import time

# to open your laptop's camera
# and store images for rock paper scissors
cap = cv2.VideoCapture(0)
cur_dir= os.getcwd() +'/' + 'training_data'  #current directory
n = int(sys.argv[2])  #number of images to be captured
folder_create = os.path.join(cur_dir, sys.argv[1]) #folder (with path) to be created
print(sys.argv[2])
print(sys.argv[1])
try:
    os.mkdir(folder_create)
except FileExistsError:
    pass
img_counter=0
while img_counter<n:
	ret, frame = cap.read()

	if not ret:
	    print("failed to capture frame")
	    cap.release()
	    cv2.destroyAllWindows()

	cv2.rectangle(frame, (100, 350), (450, 50), (0, 255, 0),15)
	cv2.imshow('capturing Gesture  : '+sys.argv[1],frame)
	k = cv2.waitKey(1)
	if k%256 == 27:#  if ESC pressed
		print("Escape hit, closing...")
		cap.release()
		cv2.destroyAllWindows()
	elif k%256 == 32: #if SPACE pressed
		img_name = "frame_{}.png".format(img_counter)
		cv2.imwrite(folder_create+'/'+sys.argv[1]+'{}.jpg'.format(img_counter),frame[50:350,100:450])
		print("{} written!".format(img_name))
		print(folder_create+'/'+sys.argv[1]+'{}.jpg Captured'.format(img_counter))
		img_counter += 1


cap.release()
cv2.destroyAllWindows()