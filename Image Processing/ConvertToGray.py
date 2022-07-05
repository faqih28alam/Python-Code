import cv2
import glob
import os

os.mkdir("/home/pi/Documents/TA/ML/dataset/ASING")
images_path = glob.glob("/home/pi/Documents/TA/ML/dataset/ASINGG/*.JPEG")

i = 0
for image in images_path:
    img = cv2.imread(image)
    gray_images = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray_images",gray_images)
    cv2.imwrite("/home/pi/Documents/TA/ML/dataset/ASING/image%02i.jpg"%i,gray_images)
    i+=1
    cv2.waitKey(600)
    cv2.destroyAllWindows()