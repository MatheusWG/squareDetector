import cv2
import numpy as np

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)

frameWidth = 640
frameHeight = 480
video_src = "20210418_134812.mp4"
cano = 200

# video_src = "20210418_134143_compressed.mp4"
if (video_src == "20210418_134143_compressed.mp4"):
    cano = 550
    cv2.createTrackbar("Threshold1", "Parameters", 180, 1000, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 313, 1000, empty)
    cv2.createTrackbar("MinLineLength", "Parameters", 101, 400, empty)
else:
    cv2.createTrackbar("Threshold1", "Parameters", 143, 1000, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 230, 1000, empty)
    cv2.createTrackbar("MinLineLength", "Parameters", 169, 400, empty)


largura = 120
cap = cv2.VideoCapture(video_src)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
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
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
            
def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

count = 0
success, img = cap.read()

while True:
    success, img = cap.read()
    count+=1


    if img is None:
        count = 0
        cap = cv2.VideoCapture(video_src)
        success, img = cap.read()

    if (count < 60):
        continue

    if (count == 140 and cano != 200):
        count = 1
        cap = cv2.VideoCapture(video_src)
        success, img = cap.read()
        continue

    imgContour = img.copy()
    imgFinal = img.copy()
    imgBlur = cv2.GaussianBlur(img, (5,5), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    # v = np.median(imgGray)
    # sigma = 0.33
    # threshold1 = int(max(300, (1.0 - sigma) * v))
    # threshold2 = int(min(255, (1.0 + sigma) * v))

    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)

    ret, binary_map = cv2.threshold(imgCanny, 127, 255, 0)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 100:
            result[labels == i + 1] = 255

    imgCanny = result
    
    kernel = np.ones((5,5), np.float32)
    imgDil = cv2.dilate(imgCanny, kernel, iterations=3)
    
    imgCannyWithoutFilter = imgCanny
    
    minLineLength = cv2.getTrackbarPos("MinLineLength", "Parameters")

    lines = cv2.HoughLinesP(image=imgDil, rho=1, theta=np.pi / 200, threshold=10, lines=np.array([]), minLineLength=minLineLength, maxLineGap=10)
    if lines is None:
        continue

    minX = minY = 10000
    maxX = maxY = 0

    # cv2.line(imgFinal, (cano, 250), (cano, 0), (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.line(imgFinal, (cano+largura, 250), (cano+largura, 0), (0, 255, 0), 2, cv2.LINE_AA)
    a, b, c = lines.shape
    for i in range(a):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]

        if (x1 > 1260 or x1 < 10):
            continue

        if (x2 > 1260 or x2 < 10):
            continue
        
        if (y1 > 1260 or y1 < 10):
            continue
        
        if (y2 > 1260 or y2 < 10):
            continue

        if (maxX < x1):
            maxX = x1

        if (minX > x1):
            minX = x1

        if (maxX < x2):
            maxX = x2

        if (minX > x2):
            minX = x2

        if (maxY < y1):
            maxY = y1

        if (minY > y1):
            minY = y1

        if (maxY < y2):
            maxY = y2

        if (minY > y2):
            minY = y2

        cv2.line(imgFinal, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    
    # if (minX > cano):
    #     cv2.line(imgFinal, (cano, 250), (cano, 0), (0, 0, 255), 2, cv2.LINE_AA)
        
    # if (minX > cano+largura):
    #     cv2.line(imgFinal, (cano+largura, 250), (cano+largura, 0), (0, 0, 255), 2, cv2.LINE_AA)

    meioCarretao = int((maxX - minX) / 2 + minX)
    meioAltura = int((maxY - minY)/2 + minY)
    # cv2.line(imgFinal, (minX, meioAltura), (maxX, meioAltura), (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(imgFinal, 'Teste', (meioCarretao, meioAltura), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
    cv2.arrowedLine(imgFinal, (meioCarretao, 50), (meioCarretao, 200), (0, 255, 0), 20, cv2.LINE_AA, 0, 0.4)

    meioCano = int(cano + largura/2)

    # cv2.putText(imgFinal, 'Teste', (meioCano-50, 260), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
    # posicaoCano = 
    valorPct = (meioCarretao - meioCano) / 100
    # print('Meio: ' + meioCarretao)
    # print('Meio Cano: ' + meioCano)
    # print('Borda Esquerda:' + minX)

    meioCarretaoRelativo = meioCarretao - minX
    posicaoCano = meioCano / meioCarretao
    if (meioCarretao > meioCano):
        posicaoCano = posicaoCano-1
    # print(max(minX, meioCano - minX))
    # print(meioCarretaoRelativo)
    print(posicaoCano)
    
    # if (maxX < cano):
    #     cv2.line(imgFinal, (cano, 250), (cano, 0), (0, 0, 255), 2, cv2.LINE_AA)
        
    # if (maxX < cano+largura):
    #     cv2.line(imgFinal, (cano+largura, 250), (cano+largura, 0), (0, 0, 255), 2, cv2.LINE_AA)

    imgStack = stackImages(0.3,([img, imgCannyWithoutFilter],
                                [imgDil, imgFinal]))
                                
    cv2.imshow("Result", imgStack)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()