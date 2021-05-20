import os
import cv2

# Contains 3 images of 3 seperate books
path = 'C:/Users/Sakshee/Pictures/image_query'

images = []  # List containing all images
classNames = []  # List containing all the corresponding class names

myList = os.listdir(path)  # contains all folders inside the path
print(myList)
print("Total Classes Detected:", len(myList))

for x in myList:
    curImg = cv2.imread(f'{path}/{x}', 0)  # importing in grayscale
    images.append(curImg)
    classNames.append(os.path.splitext(x)[0])  # appends without the extension jpg/png
print(classNames)

orb = cv2.ORB_create(nfeatures=1000)


def findDes(images):
    """
    Takes in a list of images and finds it's descriptors using orb feature detection technique (Default 500 features are detected)
    """
    desList = []

    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList


def findID(img, desList, thres=15):
    """
    Function to detect features of our webcam footage and match them with images in our directory to choose the class it most resembles
    with.
    ARGUMENTS:
    img - In this case is each frame of our webcam footage
    desList - list of descriptors of every image in our directory

    RETURNS:
    Class id of the image the web footage most resembles with.
    """
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1

    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
        print(matchList)

    except:
        pass

    if len(matchList) != 0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal


desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture(0)

while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = findID(img2, desList)
    if id != -1:
        cv2.putText(imgOriginal, classNames[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Result', imgOriginal)
    cv2.waitKey(1)
