import cv2
import numpy as np
from datetime import datetime


def median_filter(img,n):
    # expand boundary
    img2 = np.zeros((img.shape[0]+n-1,img.shape[1]+n-1))
    img2[n//2:-(n//2),n//2:-(n//2)] = img
    for j in range(n//2):
        img2[:,j] = img2[:,n-1-j]
        img2[:,-j-1] = img2[:,-n+j]
    for i in range(n//2):
        img2[i] = img2[n-1-i,:]
        img2[-i-1,:] = img2[-n+i,:]
    
    # median filter
    y = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp = img2[i:i+n,j:j+n]
            y[i,j] = np.median(tmp).astype(np.uint8)
    return y


def main():
    # initial
    video_name = 'walk.avi'
    T = 200
    video_frames = []
    Video_capture = cv2.VideoCapture(video_name)
    fps = Video_capture.get(cv2.CAP_PROP_FPS)
    ret,last_frame =  Video_capture.read()
    is_RGB = len(last_frame.shape)==3
    save_img = False
    if is_RGB:
        last_frame = np.mean(last_frame,axis=2).astype(np.uint8)
    
    #RGB to Gray
    differ1,differ2 = (None,None)
    
    # loop
    a = datetime.now()
    while True:
        ret,frame = Video_capture.read()
        if ret:
            o_frame = frame
            if is_RGB:
                frame = np.mean(frame,axis=2).astype(np.uint8)
            frame = cv2.medianBlur(frame,5)
            diff = abs(frame-last_frame)
            differ1 = (diff>=T).astype(np.uint8)
            differ1[differ1>0] = 150
            
            differ2 = cv2.medianBlur(differ1,3)#median_filter(differ1,3)
            b = datetime.now()
            # play video
            
            cv2.imshow(video_name,o_frame)
            cv2.imshow('Differ1',differ1)
            cv2.imshow('Differ2',differ2)
            cv2.waitKey(int(1000/fps))
            last_frame = frame

            # save image
            if not save_img and (b-a).seconds > 10:
                cv2.imwrite('FrameDiff_video.bmp',o_frame)
                cv2.imwrite('FrameDiff_Differ1.bmp',differ1)
                cv2.imwrite('FrameDiff_Differ2.bmp',differ2)
                save_img = True
                break
        else:
            Video_capture.release()
            cv2.destroyAllWindows()
            break

    

if __name__ == '__main__':
    main()