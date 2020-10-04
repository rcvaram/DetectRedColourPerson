# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS
# import the necessary packages
from imutils.video import VideoStream


def get_centered_contours(mask):
  # find contours
  cntrs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
  sorted_contours = sorted(cntrs, key=cv2.contourArea, reverse=True)
  filterd_contours = []
  if sorted_contours != []:
    for k in range(len(sorted_contours)):
      if cv2.contourArea(sorted_contours[k]) < 1000.0:
        filterd_contours = sorted_contours[0:k]
        return filterd_contours
  return filterd_contours


def check_red_colour_person(roi):
  hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
  # define range of blue color in HSV
  lower_red = np.array([0, 50, 50])
  upper_red = np.array([10, 255, 255])
  # Threshold the HSV image to get only blue colors
  mask = cv2.inRange(hsv, lower_red, upper_red)
  cnts = get_centered_contours(mask)
  if cnts != []:
    return True
  else:
    return False


# construct the argument parse and parse the arguments
prototxt = 'MobileNetSSD_deploy.prototxt.txt'
model = 'MobileNetSSD_deploy.caffemodel'
confidence_level = 0.8

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
  try:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
      # extract the confidence (i.e., probability) associated with
      # the prediction
      confidence = detections[0, 0, i, 2]

      # filter out weak detections by ensuring the `confidence` is
      # greater than the minimum confidence
      if confidence > confidence_level:
        # extract the index of the class label from the
        # `detections`, then compute the (x, y)-coordinates of
        # the bounding box for the object
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        roi = frame[startY:endY, startX:endX]
        # cv2.imwrite('roi_{}_{}_{}_{}.png'.format(startX,startY,endX,endY),roi)
        if check_red_colour_person(roi):
          label = "{}: {:.2f}%".format(' Red T-shirt person',
                                       confidence * 100)
          cv2.imwrite(
              'Red-T-shirt_guy_{}_{}_{}_{}.png'.format(startX, startY, endX,
                                                       endY), roi)

          cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
        else:
          cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (255, 0, 0), 2)
      y = startY - 15 if startY - 15 > 15 else startY + 15
      cv2.putText(frame, label, (startX, y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

      cv2.imshow("Frame", frame)
      key = cv2.waitKey(1) & 0xFF

      # if the `q` key was pressed, break from the loop
      if key == ord("q"):
        break

      # update the FPS counter
      fps.update()
  except Exception as e:
    print("Exception is occured")
    continue
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
