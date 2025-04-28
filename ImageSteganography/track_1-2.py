



# Python program to demonstrate 
# image steganography using OpenCV 
  
  
import cv2 
import numpy as np 
import random 
import os
import matplotlib.pyplot as plt
import scipy.ndimage as sc
  
  #%%


cale=r'D:\Facultate_1\ELECTRON\1'
imag_cale=os.path.join(cale,'track_1-2.png')
img=plt.imread(imag_cale)



  
#%%      
plt.figure('Figura 1')
plt.imshow(img,cmap='gray')
plt.title('Imaginea originala')
pixel_dim=type(img[0][0])
print(img.shape,img.dtype)
plt.show()
#%%
s=img.shape
if len(s)==3 and s[2]==3:
    img_r=img[:,:,0] *255 # *255 png
    img_g=img[:,:,1] * 255
    img_b=img[:,:,2] *255
    
    
    
    plt.figure()
    plt.subplot(1,3,1),plt.imshow(img_r,cmap='gray'),plt.title('red')
    plt.subplot(1,3,2),plt.imshow(img_g,cmap='gray'),plt.title('green')
    plt.subplot(1,3,3),plt.imshow(img_b,cmap='gray'),plt.title('blue')
    plt.show()
else:
        print('nimic')
        

#%%
#%%                               d)
def rgb2gri(img_in, format):
    img_in=img_in.astype('float')
    s=img_in.shape
    if len(s)==3 and s[2]==3:
        if format=='png':
            img_out=(0.299*img_in[:,:,0]+0.587*img_in[:,:,1]+0.114*img_in[:,:,2])*255
        elif format=='jpg':
            img_out=0.299*img_in[:,:,0]+0.587*img_in[:,:,1]+0.114*img_in[:,:,2]
        img_out=np.clip(img_out, 0,255)
        img_out=img_out.astype('uint8')
        return img_out
    else:
        print('Conversia nu a putut fi realizata deoarece imaginea de intrare nu este color!')
        return img_in
img= rgb2gri(img, 'png')
plt.figure('Imagine d')
plt.imshow(img, cmap='gray')
plt.show()
#%%

img_filtrata = sc.median_filter(img, footprint=np.ones((2,2)),mode='nearest')

plt.figure('Filtrare mediana')
plt.subplot(1,2,1), plt.imshow(img ,cmap='gray'), plt.title('Imagine originala')
plt.subplot(1,2,2), plt.imshow(img_filtrata , cmap='gray'), plt.title('Imagine prelucrată')
plt.show()

plt.figure('Imagine filtrata')
plt.imshow(img_filtrata, cmap='gray')
plt.show()
#%%
def putere(img, L, r):
    s = img.shape
    img_out = np.empty_like(img)
    img = img.astype(float)
    for i in range(0, s[0]):
        for j in range(0,s[1]):
            img_out[i,j] = (L-1)*(img[i,j] / (L-1) )**r
    img_out=np.clip(img_out , 0, 255)
    img_out=img_out.astype('uint8')
    return img_out

def contrat_pe_portiuni(img, L, a,b,Ta,Tb):
    s=img.shape
    img_out = np.empty_like(img)
    img=img.astype(float)
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            if(img[i,j]<a):
                img_out[i,j] = (Ta/a)*img[i,j]
            if(img[i,j] > a and img[i,j] < b):
                img_out[i,j] = Ta + ((Tb-Ta)/(b-a))*(img[i,j] - a) 
            if(img[i,j] >b):
                img_out[i,j] = Tb + ((L-1-Tb)/(L-1-b))*(img[i,j] - b)
    img_out = np.clip(img_out,0,255)
    img_out=img_out.astype('uint8') 
    return img_out
#%%

put = putere(img_filtrata , 255, 0.4)
contrast =  contrat_pe_portiuni(img_filtrata, 255, 80, 120, 60, 170)
plt.figure('aproape')
plt.subplot(1,2,1), plt.imshow(put,cmap='gray'), plt.title('putere')
plt.subplot(1,2,2), plt.imshow(contrast , cmap='gray'), plt.title('clpp')
plt.show()
plt.figure('Imagine filtrata')
plt.imshow(put, cmap='gray')
plt.show()


#%%
cv2.imwrite('filtru_median.png', img_filtrata)





#%%
def putere(img, L, r):
    s = img.shape
    img_out = np.empty_like(img)
    img = img.astype(float)
    for i in range(0, s[0]):
        for j in range(0,s[1]):
            img_out[i,j] = (L-1)*(img[i,j] / (L-1) )**r
    img_out=np.clip(img_out , 0, 255)
    img_out=img_out.astype('uint8')
    return img_out

def contrat_pe_portiuni(img, L, a,b,Ta,Tb):
    s=img.shape
    img_out = np.empty_like(img)
    img=img.astype(float)
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            if(img[i,j]<a):
                img_out[i,j] = (Ta/a)*img[i,j]
            if(img[i,j] > a and img[i,j] < b):
                img_out[i,j] = Ta + ((Tb-Ta)/(b-a))*(img[i,j] - a) 
            if(img[i,j] >b):
                img_out[i,j] = Tb + ((L-1-Tb)/(L-1-b))*(img[i,j] - b)
    img_out = np.clip(img_out,0,255)
    img_out=img_out.astype('uint8') 
    return img_out

def binarizare(img,L, a):
    s=img.shape
    img_out=np.ones_like(img)
    img=img.astype('uint8')
    Tb=L-1
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if(img[i,j] <a):
                img_out[i,j]=0
            else:
                img_out[i,j]=Tb
    img_out=np.clip(img_out ,0 , 255)
    img_out=img_out.astype('uint8')
    return img_out
def negativare(img, L):
    s=img.shape
    img_out=np.ones_like(img)
    img=img.astype('uint8')
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            img_out[i,j] = L-1-img[i,j]
    img_out=np.clip(img_out ,0 , 255)
    img_out=img_out.astype('uint8')
    return img_out
img_neg=negativare(img_filtrata , 256)





img_put = putere(img_filtrata, 256, 0.6)


img_cont = contrat_pe_portiuni(img_filtrata, 255, 100, 170, 50, 220) 

img_bin=binarizare(img_filtrata,255,120)



plt.figure(figsize=(10,5))
plt.subplot(1,4,1)
plt.imshow(img_bin , cmap='gray')
plt.title('binarizare')

plt.subplot(1,4,2)
plt.imshow(img_put , cmap='gray')
plt.title('Putere')


plt.subplot(1,4,3)
plt.imshow(img_cont , cmap='gray')
plt.title('Modificarea contrastului')
plt.show()

plt.subplot(1,4,4)
plt.imshow(img_neg , cmap='gray')
plt.title('Negativare')
plt.show()


#%%


#filtrare cu nucleul Sobel -> c=2
def filtrare_Sobel(img_in,c):
    w=np.array([[1,0,-1],
                [c,0,-c],
                [1,0,-1]])
    img_out=sc.convolve(img,w,mode='nearest')
    img_out=img_out.astype('uint8')
    return img_out

#filtrare cu nucleul Izotrop -> c=1.41
def filtrare_izotrop(img_in,c):
    w=np.array([[1,0,-1],
                [c,0,-c],
                [1,0,-1]])
    img_out=sc.convolve(img,w,mode='nearest')
    img_out=img_out.astype('uint8')
    return img_out

#filtrare cu nucleul Prewitt -> c=1
def filtrare_Prewitt(img_in,c):
    w=np.array([[1,0,-1],
                [c,0,-c],
                [1,0,-1]])
    img_out=sc.convolve(img,w,mode='nearest')
    img_out=img_out.astype('uint8')
    return img_out

#filtrare cu nucleul Laplacian
def filtrare_Laplacian(img_in):
    w=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    img_filt=sc.convolve(img,w,mode='nearest')
    img_filt=img_filt.astype('uint8')
    return img_filt

plt.figure('Filtrele de extragere a conturului pe imaginea test.png')
plt.subplot(2,3,1), plt.imshow(img, cmap='gray'), plt.title('Imagine originala')
plt.subplot(2,3,2), plt.imshow(img, cmap='gray'), plt.title('RGBtoGRI')
plt.subplot(2,3,3), plt.imshow(filtrare_Sobel(img,2), cmap='gray'), plt.title('Filtrare Sobel')
plt.subplot(2,3,4), plt.imshow(filtrare_izotrop(img,1.41), cmap='gray'), plt.title('Filtrare izotrop')
plt.subplot(2,3,5), plt.imshow(filtrare_Prewitt(img,1), cmap='gray'), plt.title('Filtrare Prewitt')
plt.subplot(2,3,6), plt.imshow(filtrare_Laplacian(img), cmap='gray'), plt.title('Filtrare Laplacian')
plt.show()


#%%
w1=np.array([[1,1,1,1,1]])
w2=np.array([[1],[1],[1]])
w3=np.ones((3,3))
w4=np.array([[0,1,0],[1,1,1],[0,1,0]])
w5=np.array([[1,0,0],[0,1,0],[0,0,1]])
w6=np.array([[0,0,1],[0,1,0],[1,0,0]])

def binarizare(img_ct, L, a):
    s = img_ct.shape 
    img_out = np.empty_like(img_ct)
    img_ct = img_ct.astype(float)
    Tb = L - 1
    for i in range(0,s[0]):
        for j in range(0, s[1]):
            if(img_ct[i,j] < a):
                img_out[i,j] = 0
            else:
                img_out[i,j] = Tb

    img_out = np.clip(img_out,0,255)
    img_out = img_out.astype('uint8')
    return img_out

img_modificata=binarizare(img_filtrata ,255,30)

plt.figure('Erodare imagine')
plt.subplot(3,3,1),plt.imshow(img, cmap = 'gray'),plt.title('Imagine originala')
plt.subplot(3,3,2),plt.imshow(img_filtrata , cmap = 'gray'),plt.title('Imagine filtrata')
plt.subplot(3,3,3),plt.imshow(img_modificata, cmap = 'gray'),plt.title('Imagine dupa binarizare')
x1=sc.binary_erosion(img_modificata,structure=w1)
plt.subplot(3,3,4),plt.imshow(x1, cmap = 'gray'),plt.title('Imagine cu w1')
x2=sc.binary_erosion(img_modificata,structure=w2)
plt.subplot(3,3,5),plt.imshow(x2, cmap = 'gray'),plt.title('Imagine cu w2')
x3=sc.binary_erosion(img_modificata,structure=w3)
plt.subplot(3,3,6),plt.imshow(x3, cmap = 'gray'),plt.title('Imagine cu w3')
x4=sc.binary_erosion(img_modificata,structure=w4)
plt.subplot(3,3,7),plt.imshow(x4, cmap = 'gray'),plt.title('Imagine cu w4')
x5=sc.binary_erosion(img_modificata,structure=w5)
plt.subplot(3,3,8),plt.imshow(x5, cmap = 'gray'),plt.title('Imagine cu w5')
x6=sc.binary_erosion(img_modificata,structure=w6)
plt.subplot(3,3,9),plt.imshow(x6, cmap = 'gray'),plt.title('Imagine cu w6')
plt.show()

