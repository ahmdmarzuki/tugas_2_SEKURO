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
    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, lower_bound, upper_bound)
    result = cv.bitwise_and(frame, frame, mask=mask)

    blurFrame = cv.GaussianBlur(mask, (15, 15), 0)
    
    contours, _ = cv.findContours(blurFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("gada objek")
    else:
        print("objek terdeteksi")
        
        for contour in contours:
            area = cv.contourArea(contour)
            if 1000 < area < 50000:  
                cv.drawContours(frame, [contour], -1, (0, 255, 0), 3)  

                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 

    video_resize = rescalar(frame)
    frameFileName = f"{outputFolder}/frame_{frameCount}.jpg"
    cv.imwrite(frameFileName, frame)
    
    
    
    cv.imshow("detect ball", video_resize)
            
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv.getWindowProperty('detect ball', cv.WND_PROP_VISIBLE) < 1:
        break
    
    frameCount += 1
    
video.release()
cv.destroyAllWindows



