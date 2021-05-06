import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("20210418_134143_compressed.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",23,255,empty)
cv2.createTrackbar("Threshold2","Parameters",20,1000,empty)
cv2.createTrackbar("Area","Parameters",5000,30000,empty)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

            # cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
            #             (0, 255, 0), 2)
            # cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
            #             (0, 255, 0), 2)

while True:
    success, img = cap.read()
    if img is None:
        break
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Remove hair with opening
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # Combine surrounding noise with ROI
    kernel = np.ones((6,6),np.uint8)
    dilate = cv2.dilate(opening,kernel,iterations=3)

    # Blur the image for smoother ROI
    blur = cv2.blur(dilate,(15,15))

    # Perform another OTSU threshold and search for biggest contour
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = max(contours, key=cv2.contourArea)

    # Create a new mask for the result image
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Draw the contour on the new mask and perform the bitwise operation
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    res = cv2.bitwise_and(img, img, mask=mask)

    # Display the result
    cv2.imshow('img', res)

    # imgContour = img.copy()
    # imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    # imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    
    # v = np.median(imgGray)
    # sigma = 0.33

    # #---- apply optimal Canny edge detection using the computed median----
    # lower_thresh = int(max(0, (1.0 - sigma) * v))
    # upper_thresh = int(min(255, (1.0 + sigma) * v))

    # # threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    # # threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    # imgCanny = cv2.Canny(imgGray, lower_thresh, upper_thresh)
    # kernel = np.ones((5, 5))
    # imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    # getContours(imgDil, imgContour)
    # imgStack = stackImages(0.5, ([img, imgCanny],
    #                             [imgDil, imgContour]))

    # cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
