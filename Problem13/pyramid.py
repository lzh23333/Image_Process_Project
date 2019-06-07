import cv2
import numpy as np
import matplotlib.pyplot as plt

def im_add(img1,img2):
    # 图像加减
    im = img1 + img2
    im[im>255] = 255
    im[im<0] = 0
    return im.astype(np.uint8)

def imfilter(img,kernel):
    # expand boundary
    im = np.zeros((img.shape[0]+kernel.shape[0]-1,img.shape[1]+kernel.shape[1]-1))
    b_x = kernel.shape[0]//2
    b_y = kernel.shape[1]//2
    im[b_x:-b_x,b_y:-b_y] = img

    # fill the expand boundary
    for i in range(b_x):
        im[:,b_x-i-1] = im[:,b_x+i+1]
        im[:,-b_x+i] = im[:,-b_x-i-2]
    
    for j in range(b_y):
        im[b_y-j-1,:] = im[b_y+j+1,:]
        im[-b_y+j,:] = im[-b_y-j-2,:]
    
    # covn
    y = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            y[i,j] = np.sum(kernel*im[i:i+kernel.shape[0],j:j+kernel.shape[1]])
    
    
    y[y<0] = 0
    y[y>255] = 255
    return y.astype(np.uint8)


def down_sample(img):
    #下采样
    r,c = img.shape
    im = np.zeros((r//2,c//2))
    for i in range(r//2):
        for j in range(c//2):
            im[i,j] = img[2*i,2*j]
    return im

def up_sample(img):
    #上采样
    r,c = img.shape
    im = np.zeros((2*r,2*c))
    for i in range(0,2*r,2):
        for j in range(0,2*c,2):
            im[i,j] = img[i//2,j//2]
    return im


def gaussian_pyramid(img,kernel,T):
    # img为原图像
    # kernel为卷积核
    # T为高斯金字塔数
    # 返回高斯金字塔
    im_list = [img]
    for i in range(T):
        im_list.append(down_sample(imfilter(im_list[-1],kernel)))
    return im_list

def laplace_pyramid(gaussian_list,kernel):
    #gaussian_list为高斯金字塔列表
    #函数返回拉普拉斯金字塔列表
    im_list = []
    for i in range(len(gaussian_list)-1):
        gauss1 = gaussian_list[i]
        gauss2 = gaussian_list[i+1]
        im_list.append(im_add(gauss1,-imfilter(up_sample(gauss2),kernel)))
    return im_list

def reconstruct_img(gaussian_img,laplace_list,kernel):
    # gaussian_img为最后一幅高斯图像
    # laplace_list 为拉普拉斯金字塔
    # kernel为卷积核
    img = None
    for i in range(len(laplace_list)):
        l_im = laplace_list[-i-1]
        gaussian_img = im_add(imfilter(up_sample(gaussian_img),kernel),l_im)
    return gaussian_img

def main():
    # init 
    kernel1 = 1/8*np.array([1,2,2,2,1])
    kernel2 = 1/16*np.array([1,4,6,4,1])
    K1 = np.multiply(kernel1.reshape(-1,1),kernel1)
    K2 = np.multiply(kernel2.reshape(-1,1),kernel2)
    img = cv2.imread('Lena.bmp',0)
    T = 3

    K = K1

    # make pyramids
    gaussian_list = gaussian_pyramid(img,K,T)
    laplace_list = laplace_pyramid(gaussian_list,4*K)
    img2 = reconstruct_img(gaussian_list[-1],laplace_list,4*K)


    # Q1
    for i in range(T+1):
        #cv2.imwrite('Gaussian_pyramid'+str(i)+'.bmp',gaussian_list[i])
        pass

    # Q2
    for i in range(T):
        #cv2.imwrite('laplace_pyramid'+str(i)+'.bmp',laplace_list[i])
        pass


    # Q3
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title('origin')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img2,cmap='gray')
    plt.title('reconstruct')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()





