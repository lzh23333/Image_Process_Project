import numpy as np
import cv2

class MOG:
    K = 5
    sigma = 20
    def __init__(self,alpha,T,frame):
        self.alpha = alpha
        self.T = T
        self.distribution = []
        self.r,self.c = (frame.shape[0],frame.shape[1])
        self.isRGB = len(frame.shape)==3
        self.sample_num = 1
        for i in range(frame.shape[0]):
            self.distribution.append([])
            for j in range(frame.shape[1]):
                K_dis = [[1,frame[i,j],sigma]]
                for k in range(1,K):
                    K_dis.append([0,np.zeros((1,3)),sigma])
                self.distribution[i].append(K_dis)
        print('init distribution done!')

    def detect(self,frame):
        # return a tuple(background,foreground)
        background = np.zeros(frame.shape)
        foreground = np.zeros((self.r,self.c))
        self.sample_num += 1

        for i in range(self.r):
            for j in range(self.c):
                K_dis = self.distribution[i,j]
                B = self.backdis(K_dis)
                match = match_pixel(frame[i,j],K_dis)
                is_fore = True
                if match is not None:
                    if match[0] <= B:
                        # which means it's a foreground pixel
                        is_fore = False
                    self.distribution[i,j] = self.update_distribution(frame[i,j],K_dis)
                else:
                    # no match, need to replace one distribution
                    new_dis = [1/self.sample_num,frame[i,j],self.sigma]
                    self.distribution[i,j,find_least_prob(frame[i,j],self.distribution[i,j])] = new_dis
                    self.update_weight(self.distribution[i,j])
                
                if is_fore:
                    foreground[i,j] = 150
                    # choose max weight distribution's mu as background pixel
                    background[i,j] = self.distribution[i,j,0,1]
                else :
                    background[i,j] = 0
                    background[i,j] = frame[i,j]

        return (background,foreground)

    def match_pixel(self,x,K_dis):
        pass

    def update_distribution(self,x,K_dis):
        pass
    
    def sort_distribution(self,dis):
        # sort the distribution using value w/sigma
        return dis.sort(reverse=True,key=lambda x: x[0]/x[2])

    def find_least_prob(self,x,K_dis):
        # return idx of least proba dis
        pass
    
    def update_weight(self,K_dis):
        total_weight = 0
        for dis in K_dis:
            total_weight += dis[0]
        for i in range(len(K_dis)):
            K_dis[i,0] = K_dis[i,0]/total_weight
        return K_dis

    def background(self,K_dis):
        a = 0
        for B in range(K):
            a += K_dis[B,0]
            if a>T:
                return B
        