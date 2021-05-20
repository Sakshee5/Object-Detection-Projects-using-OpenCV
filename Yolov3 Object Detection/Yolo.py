"""
Yolov3 model has been trained on the COCO dataset (80 classes of objects). The pretrained yolo configuration used has image dimensions 608 x 608. File coco.names contains the names of the 80 COCO classes.

NON-MAX SUPPRESSION:
To supress multiple detections for the same object. We reduce the nmsThreshold value for more aggressive suppression. Sometimes what happens is that more than one box points to the same object. In this case, instead of one detection we would have 2 detections, even though in reality we just have one object

To counter this problem we will use the Non Max Suppression. In the simplest term the NMS eliminates the overlapping boxes. It
finds the overlapping boxes and then based on their confidence it will pick the max confidence box and supress all the non max boxes.
So we will use the builtin NMSBoxes function

DETECTION CONFIDENCE: 
Confidence value of whether an object is present in a detected bounding box. (This is the first output parameter in the output vector.)

NETWORK INPUT
We cannot send our image form the camera directly to the network. It has to be in a certain format called blob. We can use the blobFromImage
function which creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels. We will keep all the values at default.

NETWORK OUTPUT
Since the Yolo3 Architecture has 3 output layers we will find their names so we can get their outputs.
"""
import cv2
import numpy as np

video = cv2.VideoCapture(0)
video.set(3, 640)
video.set(4, 480)

whT = 608
confThreshold = 0.5  # so that we can save bounding boxes which have relevant objects i.e have good confidence number
nmsThreshold = 0.3  # Non-Max suppression parameter

# Here, we will first collect the names of our classes in a list
classesFile = "C:/Users/Sakshee/PycharmProjects/OpenCV-Projects/coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
print(len(classNames))

modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# to filter out the low confidence ones and create a list of relevant boxes that contain objects.
def findObjects(outputs, img):
    """
    Function to filter out bounding boxes with low confidence scores and to make a list of relevant bounding boxes detected.
    """
    hT, wT, cT = img.shape
    bbox = []  # bounding box corner points
    classIds = []  # class id with the highest confidence
    confs = []  # confidence value of the highest class

    for output in outputs:  # output is the prediction of each layer
        for detection in output:  # each box in the output is called detection
            scores = detection[
                     5:]  # remove the first 5 values from the detection since they are bbox co-ordinates and confidence
            classId = np.argmax(scores)  # outputs index values of our max value
            confidence = scores[classId]

            if confidence > confThreshold:
                w, h = int(detection[2] * wT), int(
                    detection[3] * hT)  # multipying by height and width since values given are percentages
                x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)
                # subtracting height and width because the output values are center values

                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold,
                               nmsThreshold)  # this will tell us which bounding boxes are relevant
    # print(indices)                                          # this gives us [[value]], [[value]], [[value]], [[value]], .....85 values

    # Drawing bounding boxes
    for i in indices:
        i = i[0]
        print(i)
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x, y, w, h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = video.read()

    # converting image into a format that the network accepts i.e. the blob format
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layersNames = net.getLayerNames()
    print(layersNames)

    # Since we need only the output layers we extract them (yolo has 3 output layers)
    # print(net.getUnconnectedOutLayers())       # we observe that we get the index of the outputs

    # so we take the indexes and refer them back to the layer names so that we can extract the names from these indices
    # Since we use 0 as the first element we have to subtract 1 from the indices we get from the getUnconnectedOutLayers function
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    print(outputNames)
    print(net.getUnconnectedOutLayers())

    outputs = net.forward(outputNames)
    # print(len(outputs))                # we get 3 different outputs
    # print(type(outputs))               # we observe it's of type list
    # print(type(outputs[0]))            # we observe that it's a numpy array
    # print(outputs[0].shape)            # (300, 85)
    # print(outputs[1].shape)            # (1200, 85)
    # print(outputs[2].shape)            # (4800, 85)
    # print(outputs[0][0])               # we will get an array of 85 values

    findObjects(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

"""
In the output layer shape the 300, 1200, and the 4800 are the number of boxes we get from the respective output layers. But what is the value of 85? If you have to define a box you can simply define it with the center position cx,cy and the diagonal corner w,h.
This means we only need 4 values (cx, cy, w, h), so our arrays should be 300×4, 1200×4, and 4800×4 .Then what are the rest of 81 values?

The fifth value is the confidence which tells us how likely is it that we have an object in this box. The rest of the 80 values correspond
to each class confidence. So if there was a car in the image then, 5 + 3 = 8th element would show a high confidence value e.g. 0.7
(car is the 3rd element in the list of coco names).
"""