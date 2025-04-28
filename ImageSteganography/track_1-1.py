import cv2
import numpy as np
import matplotlib.pyplot as plt

# Citim imaginile
imgA = cv2.imread(r'D:\Facultate_1\ELECTRON\1\track_1-1a.png')
imgB = cv2.imread(r'D:\Facultate_1\ELECTRON\1\track_1-1b.png')
print(imgA.shape,imgA.dtype)
print(imgB.shape,imgB.dtype)
#%%
# Diferenta pixel cu pixel
diffBA = cv2.absdiff(imgB, imgA)*800
diffAB = cv2.absdiff(imgA, imgB)*500
# Afisezi rezultatul

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(diffBA, cv2.COLOR_BGR2RGB))
plt.title('Diferenta B - A')

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(diffAB, cv2.COLOR_BGR2RGB))
plt.title('Diferenta A - B')
plt.show()




#%%

# Separi canalele
colors = ('b', 'g', 'r')  # ordinea BGR în OpenCV

plt.figure()
for i, color in enumerate(colors):
    hist = cv2.calcHist([diffBA], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.title('Histograma RGB   B-A')
plt.xlabel('Intensitate Pixel')
plt.ylabel('Numar Pixeli')
plt.show()

#%%

# Separi canalele
colors = ('b', 'g', 'r')  # ordinea BGR în OpenCV

plt.figure()
for i, color in enumerate(colors):
    hist = cv2.calcHist([diffAB], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.title('Histograma RGB   A-B')
plt.xlabel('Intensitate Pixel')
plt.ylabel('Numar Pixeli')
plt.show()



#%%
# XOR pixel cu pixel
xor = cv2.bitwise_xor(imgA, imgB)

# Afisezi rezultatul
plt.imshow(cv2.cvtColor(xor, cv2.COLOR_BGR2RGB))
plt.title('XOR intre A si B')
plt.show()

#%%

# Separi canalele
colors = ('b', 'g', 'r')  # ordinea BGR în OpenCV

plt.figure()
for i, color in enumerate(colors):
    hist = cv2.calcHist([diffBA], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.title('Histograma RGB')
plt.xlabel('Intensitate Pixel')
plt.ylabel('Numar Pixeli')
plt.show()






#%%

def extract_strings(file_path, min_length=4):
    with open(file_path, "rb") as f:
        data = f.read()

    result = []
    current_string = ""

    for byte in data:
        if 32 <= byte <= 126:  # caractere ASCII printabile
            current_string += chr(byte)
        else:
            if len(current_string) >= min_length:
                result.append(current_string)
            current_string = ""

    if len(current_string) >= min_length:
        result.append(current_string)

    return result

# Exemplu de utilizare
file = r'D:\Facultate_1\ELECTRON\1\track_1-1a.png'
strings = extract_strings(file)

for s in strings:
    print(s)