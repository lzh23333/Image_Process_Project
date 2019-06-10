import cv2
import numpy as np

def float2uint(img):
    img[img>255] = 255
    img[img<0] = 0
    return img.astype(np.uint8)

def main():
    # initial parameters
    video_name = 'walk.avi'
    Video_capture = cv2.VideoCapture(video_name)
    fps = Video_capture.get(cv2.CAP_PROP_FPS)
    N = 50
    alpha = 0.05
    T = 70
    frame1 = None
    background = None
    foreground = None
    save_img = False
    init = False
    idx = 0

    while True:
        ret,frame = Video_capture.read()
        
        if ret:
            frame1 = frame  # for save img
            frame = frame.astype(np.float32)
            if not init:
                idx+=1
                # init background
                if idx < N:
                    if background is None:
                        background = frame
                    else:
                        background = background+frame
                elif idx == N:
                    background = background/N
                    init = True
                    print('background init done')
            else:
                # update background
                background = (1-alpha)*background + alpha*frame
                # foreground detect
                foreground = np.linalg.norm((frame-background),axis=2)
                foreground[foreground<T] = 0
                foreground[foreground>=T] = 150
                foreground = foreground.astype(np.uint8)

                # play video
                cv2.imshow('background',float2uint(background))
                cv2.imshow('foreground',foreground)
            # show image
            cv2.imshow(video_name,frame.astype(np.uint8))
            
            cv2.waitKey(int(1000/fps))
        else:
            Video_capture.release()
            cv2.destroyAllWindows()
            break
    cv2.imwrite('AverageB.bmp',float2uint(background))
    cv2.imwrite('AverageS.bmp',foreground)
    cv2.imwrite('AverageVideo.bmp',frame1.astype(np.uint8))

if __name__ == "__main__":
    main()