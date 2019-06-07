import cv2
import numpy as np
import matplotlib.pyplot as plt

def map2RGB(f):
    # 灰度映射到RGB函数
    R = f
    G = 255-4/255*(f-127.5)*(f-127.5)
    B = 255-f
    img = np.array([B,G,R]).transpose(1,2,0)
    return img.astype(np.uint8)

def HSI(img):
    ## RGB-->HSI
    img = img.astype(np.float32)
    print(img.dtype)
    I = np.mean(img,axis=2)
    S = 1-3*np.min(img,axis=2)/np.sum(img,axis=2)
    H = np.zeros(I.shape)

    def calc_H(channel):
        [B,G,R] = channel
        H = np.arccos((2*R-G-B)/2/np.sqrt((R-G)**2+(R-B)*(G-B)))
        if B>G:
            H = 2*np.pi-H
        return H
    
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            H[i,j] = calc_H(img[i,j,:])

    return (H,S,I)

def main():
    file_name = 'Lena_color.bmp'
    img = cv2.imread(file_name)
    print(img.shape)
    H,S,f = HSI(img)
    print(f.shape)
    img = map2RGB(f)
    print(img.shape)


    cv2.imwrite('transform.bmp',img)
    cv2.imshow('1',img)
    cv2.waitKey(0)

    

if __name__ == '__main__':
    main()
