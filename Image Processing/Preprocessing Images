#https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import image as mpimg

#alamat gambar wajah grayscale "/home/pi/Documents/TA/DATASET/Dataset_Gambar/TRAINING/FAQ/FAQ (1).jpg"
#path = "/home/pi/Documents/TA/DATASET/Dataset_Gambar/TRAINING/FAQ/FAQ (1).jpg"
#alamat gambar wajah rgb "/home/pi/Documents/TA/ML/dataset_terbaru_jpg/FAQIH/FAQ.1.jpg"
path  = "/home/pi/Documents/TA/ML/dataset_terbaru_jpg/FAQIH/FAQ.1.jpg"
path2 = "/home/pi/Documents/COBA_COBA/Result/test01.jpg"
#alamat hasil konversi preprocessing
path_result = "/home/pi/Documents/COBA_COBA/Result"

#membaca/load gambar
img = cv2.imread(path)
img2 = cv2.imread(path2)

#konversi gambar ke grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#img  = cv2.imread(path,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(path2,cv2.COLOR_BGR2GRAY)
#img = cv2.imread(path, 0)
#membaca gambar "yang telah digrayscale"
#img = cv2.imread(path,cv2.COLOR_BGR2GRAY)

#merubah ukuran gambar/resize
#img = cv2.resize(img, (200,200))

#merubah ukuran gambar/resize + merubah bentuk gambar menjadi satu dimensi
img  = cv2.resize(img, (200,200)).flatten()
img2 = cv2.resize(img2, (200,200)).flatten()

#menampilkan gambar
cv2.imshow("Gambar Wajah",img)
# plt.imshow(img,cmap='Greys')
# plt.show()

#read image using matplotlib lib and convert to numpy array 
# img_np = np.array(mpimg.imread(path))
# img_np = img_np.setflags(write = 1)
# print("Image shape", img_np.shape)
# plt.imshow(img_np)
# plt.show

#holds a tuple (rows(width), columns(heihgt), channels) 
#print(img.shape)
print("img shape  :" + str(img.shape))
print("img2 shape :" + str(img2.shape))


print("list array  :" + str(img))
print("list array2 :" + str(img2))

print("jenis data  :" + str(type(img)))
print("jenis data2 :" + str(type(img2)))

print("hasil img   :" + str(img.sum()))
print("hasil img2  :" + str(img2.sum()))

selisih = (img.sum())-(img2.sum())

print("selisih img & img2: " + str(selisih))

#menyimpan file gambar
# i = 0
# cv2.imwrite(path_result + "/test%02i.jpg"%i,img)
# i += 1

#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)   
#closing all open windows 
cv2.destroyAllWindows()
