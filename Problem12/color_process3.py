import cv2
import numpy as np
import matplotlib.pyplot as plt

def HSI(img):
    ## RGB-->HSI
    img = img.astype(np.float32)
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

def HSI2RGB(channel):
    # 将HSI单通道转为RGB通道
    H,S,I = channel
    R,G,B = 0,0,0

    if H < 2/3*np.pi:
        B = I*(1-S)
        R = I*(1+(S*np.cos(H))/np.cos(np.pi/3-H))
        G = 3*I-B-R
    elif 2/3*np.pi <= H and H < 4/3*np.pi:
        R = I*(1-S)
        G = I*(1+(S*np.cos(H-2/3*np.pi))/np.cos(np.pi-H))
        B = 3*I-G-R
    elif 4/3*np.pi <= H and H <= 2*np.pi:
        G = I*(1-S)
        B = I*(1+(S*np.cos(H-4/3*np.pi))/np.cos(5/3*np.pi-H))
        R = 3*I-B-G
    
    return (R,G,B)

def RGB(H,S,I):
    # 根据HSI恢复RGB
    R = np.zeros(H.shape)
    G = np.zeros(H.shape)
    B = np.zeros(H.shape)

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            R[i,j],G[i,j],B[i,j] = HSI2RGB((H[i,j],S[i,j],I[i,j]))
    
    return (R,G,B)

def main():
    file_name = 'Lena_color.bmp'
    img = cv2.imread(file_name)
    print(img.shape)
    H,S,I = HSI(img)

    b,g,r = cv2.split(img)


    R0,G0,B0 = RGB(H,S,I)
    img0 = cv2.merge([R0,G0,B0]).astype(np.uint8)

    S1 = 1.1*S
    S1[S1>1] = 1
    R1,G1,B1 = RGB(H,S1,I)
    img1 = cv2.merge([R1,G1,B1]).astype(np.uint8)

    S2 = 0.6*S
    R2,G2,B2 = RGB(H,S2,I)
    img2 = cv2.merge([R2,G2,B2]).astype(np.uint8)

    plt.subplot(2,2,1)
    plt.imshow(cv2.merge([r,g,b]))
    plt.axis('off')
    plt.title('origin')
    plt.subplot(2,2,2)
    plt.imshow(img0)
    plt.axis('off')
    plt.title('HSI->RGB')
    plt.subplot(2,2,3)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('1.1 S')
    plt.subplot(2,2,4)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('0.6 S')

    
    plt.show()


    
    



if __name__ == '__main__':
    main()
