# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2
import urllib
import pymongo
from twilio.rest import Client
import os

def refreshLine(line, collection):
    print("[INFO] refreshing reservation queue...")

    # query the database for all reservations
    documents = collection.find({})
    for doc in documents:
        # send an sms message to any new reservations
        if doc['_id'] not in [x['_id'] for x in line]:
            line.append(doc)
            sendSMS(doc['phone'], "Thank you for getting in line. You are #" + str(collection.count()) +" in line.")

    # sort by datetime created
    line.sort(key=lambda x: x['createdAt'])

def sendSMS(number, body):
    number = str(int(number))
    print(number)
    message = client.messages.create(
        to="+1" + number, 
        from_="+17262684714",
        body=body)

# create the database URI from the .env data
URI = "mongodb+srv://%s:%s@cluster0.fmpvk.mongodb.net/customers?retryWrites=true&w=majority" \
    % (os.getenv("USER"), urllib.parse.quote_plus(str(os.getenv("PASS"))))

# establish twilio connection
print("[INFO] establishing connection to twilio...")
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token  = os.getenv('TWILIO_AUTH_TOKEN')

client = Client(account_sid, auth_token)

# establish database connection
print("[INFO] establishing connection to database...")

conn = pymongo.MongoClient(URI)
db = conn.get_database('customers')
occupancies = db.occupancies
reservations = db.datas

# currently we are hard coding the only available location
LOC = 'marvil_home'
data = occupancies.find_one({'location':LOC})
occupancy = data['occupancy']
maxOccupancy = data['max_occupancy']

# generate the list of reservations from the database
line = []
refreshLine(line, reservations)

# load the object detection model
print("[INFO] loading model...")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# start the video input stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize our window
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

width, height = None, None
detectionFrames, refreshFrames = 0, 0
occupancyChanged = False

fps = FPS().start()

# process the webcam feed frame by frame
while True:
    frame = vs.read()

    # cap frame size at 500 to improve performance
    frame = imutils.resize(frame, width=500)
    #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = frame

    if width is None and height is None:
        (height, width) = frame.shape[:2]

    status = "Waiting"
    rects = []

    # refresh the reservations list every 120 frames
    if refreshFrames > 120:
        refreshLine(line, reservations)
        refreshFrames = 0

    # run the object detection every 30 frames
    if detectionFrames > 30:
        status = "Detecting"
        trackers = []

        # convert the frame to a blob and pass the blob through the network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum confidence
            if confidence > 0.4:
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding box 
                # coordinates and then start the dlib correlation tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)

        detectionFrames = 0
    else:
        # loop over the trackers
        for tracker in trackers:
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX, startY = int(pos.left()), int(pos.top())
            endX, endY = int(pos.right()), int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # add a visual indicator of the store entrance
    cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 255), 2)

    # update the centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # get the trackable object by id
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            # get the difference between current and previous centroids
            x = [c[0] for c in to.centroids]
            direction = centroid[0] - np.mean(x)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if dir is negative and the line has been crossed decrement occupancy
                if direction > 0 and centroid[0] < width // 2 and occupancy > 0:
                    occupancy -= 1
                    to.counted = True
                    occupancyChanged = True

                    # when someone exits, notify the next 3 people in line
                    if len(line) >= 1:
                        sendSMS(line[0]['phone'], "You may now enter the store. Please arrive within five minutes to avoid losing your reservation.")
                    if len(line) >= 2:
                        sendSMS(line[1]['phone'], "You are now next in line. Please prepare to enter.")
                    if len(line) >= 3:
                        sendSMS(line[2]['phone'], "You are now second in line.")
                # if dir is positive and the line has been crossed increment occupancy
                elif direction < 0 and centroid[0] > width // 2:
                    occupancy += 1
                    to.counted = True
                    occupancyChanged = True

                    # if the line is not empty
                    if len(line):
                        # send a confirmation that they have entered
                        sendSMS(line[0]['phone'], "The OTIS vision system has checked you in. Thank you for waiting and enjoy your stay. \
                            If you have not entered and think this is a mistake, please contact 571-442-0642 for assistance.")

                        # remove from the line and the db
                        reservations.delete_one({'_id': line[0]['_id']})
                        line.pop(0)
        
        # store the trackable object in the dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)    

    # display information
    text = "Occupancy: " + str(occupancy) + "/" + str(maxOccupancy)
    cv2.putText(frame, text, (5, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "Status: " + status, (5, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow('window', frame)

    # update the database
    if occupancyChanged:
        occupancies.update_one({'location':'marvil_home'}, {'$set': {'occupancy':occupancy}})
        occupancyChanged = False

    # break on escape
    if cv2.waitKey(1) & 0xFF == 27:
        break

    refreshFrames += 1
    detectionFrames += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# clean up
vs.stop()
cv2.destroyAllWindows()
