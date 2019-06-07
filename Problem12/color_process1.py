import cv2
import numpy as np


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
    H,S,I = HSI(img)
    cv2.imwrite('B.bmp',img[:,:,0])
    cv2.imwrite('G.bmp',img[:,:,1])
    cv2.imwrite('R.bmp',img[:,:,2])
    cv2.imwrite('H.bmp',(H/2/np.pi*255).astype(np.uint8))
    cv2.imwrite('S.bmp',(S*255).astype(np.uint8))
    cv2.imwrite('I.bmp',I.astype(np.uint8))


if __name__ == '__main__':
    main()
