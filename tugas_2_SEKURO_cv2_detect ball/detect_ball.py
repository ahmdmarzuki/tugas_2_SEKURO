import cv2 as cv
import numpy as np
import os

video = cv.VideoCapture("assets/myvid.mp4")
# video = cv.VideoCapture(0)
frameCount = 0

outputFolder = "video_parse_to_image"
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
    
lower_bound = np.array([170, 100, 50])
upper_bound = np.array([180, 255, 255])


    
def rescalar(frame, scale=.4):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimention = (width, height)
    
    rescalar = cv.resize(frame, dimention, interpolation=cv.INTER_AREA)
    
    return rescalar

while True:
    ret, frame = video.read()
    if not ret or frame is None:
        break
    
    
    # konversi warna ke hsv
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # masking
    mask = cv.inRange(hsv, lower_bound, upper_bound)
    result = cv.bitwise_and(frame, frame, mask=mask)
    
    # blur
    blurFrame = cv.GaussianBlur(mask, (15, 15), 0)
    
    contours, _ = cv.findContours(blurFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("gada contours")
    else:
        print("adaaa")
        
        for contour in contours:
            area = cv.contourArea(contour)
            if 1000 < area < 50000:  
                cv.drawContours(frame, [contour], -1, (0, 255, 0), 3)  

                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 

    
    # # detect circle
    # circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, dp=1.2, minDist=100,
    #                           param1=50, param2=50, minRadius=10, maxRadius=100)
    
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         # Gambar lingkaran pada bola yang terdeteksi
    #         cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 3)
    #         cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    video_resize = rescalar(frame)
    frameFileName = f"{outputFolder}/frame_{frameCount}.jpg"
    cv.imwrite(frameFileName, frame)
    
    
    
    cv.imshow("detect ball", video_resize)
            
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    frameCount += 1
    
video.release()
cv.destroyAllWindows



