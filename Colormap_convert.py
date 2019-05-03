import cv2
from numpy import *
import numpy as np
import math
import copy

##
#define image filter function
def image_filter(img_input,filter_kernel):
    
    size_img_input=img_input.shape
    kernel_size=filter_kernel.shape
    decenter=int((kernel_size[1]-1)/2)
    rgb_gray=len(size_img_input)
    if rgb_gray==3: 
        new_img=zeros((size_img_input[0],size_img_input[1],size_img_input[2]))
        rgb_channel=size_img_input[2]
        for c in range(0,rgb_channel):
            img_ex=img_input[:,:,c]
            for i in range(0,size_img_input[0]):
                for j in range(0,size_img_input[1]):
                    H=zeros((kernel_size[0],kernel_size[1]))
                    G=zeros((kernel_size[0],kernel_size[1]))
                    for n in range(-decenter,decenter+1):
                        for m in range(-decenter,decenter+1):
                            if i+n in range(0,size_img_input[0]) and j+m in range(0,size_img_input[1]):
                                H[decenter+n,decenter+m]=img_ex[i+n,j+m]
                            else:
                                H[decenter+n,decenter+m]=0
                    for x in range(-decenter,decenter+1):
                        for y in range(-decenter,decenter+1):
                            G[decenter+x,decenter+y]=H[decenter+x,decenter+y]*filter_kernel[decenter-x,decenter-y]
                    new_img[i,j,c]=sum(G)
    else:
        new_img=zeros((size_img_input[0],size_img_input[1]))
        for i in range(0,size_img_input[0]):
                for j in range(0,size_img_input[1]):
                    H=zeros((kernel_size[0],kernel_size[1]))
                    G=zeros((kernel_size[0],kernel_size[1]))
                    for n in range(-decenter,decenter+1):
                        for m in range(-decenter,decenter+1):
                            if i+n in range(0,size_img_input[0]) and j+m in range(0,size_img_input[1]):
                                H[decenter+n,decenter+m]=img_input[i+n,j+m]
                            else:
                                H[decenter+n,decenter+m]=0
                    for x in range(-decenter,decenter+1):
                        for y in range(-decenter,decenter+1):
                            G[decenter+x,decenter+y]=H[decenter+x,decenter+y]*filter_kernel[decenter-x,decenter-y]
                    new_img[i,j]=sum(G)
        
    img_new_adjusted=new_img.clip(0, 255)
    img_new=np.rint(img_new_adjusted).astype('uint8')
    return img_new

##
#define function to generate gaussian kernel
def gaus_kernel(sigma,pixel_num):
    Gau=zeros((pixel_num,pixel_num))
    hw=1/2*(pixel_num-1)
    ##hw=half width, being calculated to adjust the center position
    ##
    for x_g in range(0,pixel_num):
        for y_g in range(0,pixel_num):
            Gau[x_g,y_g]=(1/(2*math.pi*sigma*sigma))*exp(-((x_g-hw)*(x_g-hw)+(y_g-hw)*(y_g-hw))/(2*sigma*sigma))
    sum_Gau=sum(Gau)
    for x_g in range(0,pixel_num):
        for y_g in range(0,pixel_num):
            Gau[x_g,y_g]=Gau[x_g,y_g]/sum_Gau
    return Gau

##
##define sampling function
def sampling(image_original,ratio):
    size_original=image_original.shape
    length=size_original[0]
    width=size_original[1]
    length_adj=math.floor(length/ratio)
    width_adj=math.floor(width/ratio)
    rgb_or_gray=len(size_original)
    if rgb_or_gray==3:
        sampled_img=zeros((length_adj,width_adj,size_original[2]))
        for colorchannel in range(0,size_original[2]):
            for x_adj in range(0,length_adj):
                for y_adj in range(0,width_adj):
                    sampled_img[x_adj,y_adj,colorchannel]=np.mean(image_original[ratio*x_adj:ratio*(x_adj+1),ratio*y_adj:ratio*(y_adj+1),colorchannel])
    else:
        sampled_img=zeros((length_adj,width_adj))
        for x_adj in range(0,length_adj):
                for y_adj in range(0,width_adj):
                    sampled_img[x_adj,y_adj]=np.mean(image_original[ratio*x_adj:ratio*(x_adj+1),ratio*y_adj:ratio*(y_adj+1)])

    sampled_img_output=np.rint(sampled_img).astype('uint8')
    return sampled_img_output
def kmeans_img(image,k):
    img=image.astype('float64')
    l=img.shape[0]
    w=img.shape[1]
    dataset=[]
    locset=[]
    for i_l in range(0,l):
        for i_w in range (0,w):
            info=[img[i_l,i_w,0],img[i_l,i_w,1],img[i_l,i_w,2]]
            loc=[i_l,i_w]
            dataset.append(info)
            locset.append(loc)
    ind_r=np.random.randint(0,len(dataset),k)
    b_ran=[dataset[x][0] for x in ind_r]
    g_ran=[dataset[x][1] for x in ind_r]
    r_ran=[dataset[x][2] for x in ind_r]
    b_tem=[0]*k
    g_tem=[0]*k
    r_tem=[0]*k
    comp=0
    while comp==0:
        cluster_flag=np.zeros([len(dataset),1]).astype(int)
        for n in range(0,len(dataset)):
            dis=[]
            for m in range(0,k):
                d=((dataset[n][0]-b_ran[m])**2+(dataset[n][1]-g_ran[m])**2+(dataset[n][2]-r_ran[m])**2)**0.5
                dis.append(d)
            flag=np.argmin(dis)
            cluster_flag[n]=flag
        for i in range(0,k):
            index=[]
            for j in range(0,len(dataset)):
                if cluster_flag[j]==i:
                    index.append(j)
            b_tem[i],g_tem[i],r_tem[i]=np.mean([dataset[ii] for ii in index],axis=0)
        if np.all([np.all(np.round(b_ran)==np.round(b_tem)),np.all(np.round(g_ran)==np.round(g_tem)),np.all(np.round(r_ran)==np.round(r_tem))]):
            comp=1
        else:
            b_ran=copy.deepcopy(b_tem)
            g_ran=copy.deepcopy(g_tem)
            r_ran=copy.deepcopy(r_tem)
        
    img_new=np.zeros([l,w,3])
    for num in range(0,len(dataset)):
        img_new[locset[num][0],locset[num][1],:]=[b_ran[cluster_flag[num][0]],g_ran[cluster_flag[num][0]],r_ran[cluster_flag[num][0]]]    
    img_new=img_new.astype('uint8')
    
    return img_new
    
img_o=cv2.imread("/Users/yangyunchen/Dropbox/Python/forZH/Vegas.jpg")
img_grey_o=cv2.imread("/Users/yangyunchen/Dropbox/Python/forZH/Vegas.jpg",0)
#cv2.imshow('original img',img_o)
img=sampling(img_o,4)
Gau_f=gaus_kernel(sqrt(2),7)
img=image_filter(img,Gau_f)

##
#clustering of image
img_clust=kmeans_img(img,18)
img_edge=cv2.Canny(img,100,120)
i=np.where(img_edge==255)
for nc in range(0,len(i[0])):
    img_edge[i[0][nc]-1:i[0][nc]+2,i[1][nc]-1:i[1][nc]+2]=255
ind=np.where(img_edge==255)
img_clust[:,:,0][ind]=0
img_clust[:,:,1][ind]=0
img_clust[:,:,2][ind]=0
cv2.imshow('img_clust',img_clust)
cv2.imwrite('/Users/yangyunchen/Dropbox/Python/forZH/Vegas_cluster_k18_sketch.jpg',img_clust)