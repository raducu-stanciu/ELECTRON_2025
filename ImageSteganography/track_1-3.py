

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
imag_cale=os.path.join(cale,'track_1-3.png')
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
        
cv2.imwrite('red.png', img_r)        
cv2.imwrite('green.png', img_g)
cv2.imwrite('blue.png', img_b)

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