#https://www.youtube.com/watch?v=TKfTKA5EmmY&t=130s
#https://stackoverflow.com/questions/70945386/how-to-load-custom-image-dataset-to-x-train

##################importing libraries
import sklearn
from tqdm import tqdm
import os
import cv2
import numpy as np

##################preparing dataset
#set a variable to the dictionary of your class names:
classes = {'ASING':0,'FAQ':1,'FZN':2,'GGZ':3,'LDY':4,'NGH':5,'NTA':6}

ASING_DIR='/home/pi/Documents/TA/Dataset_Gambar/TRAINING/ASING'
FAQ_DIR='/home/pi/Documents/TA/Dataset_Gambar/TRAINING/FAQ'
FZN_DIR='/home/pi/Documents/TA/Dataset_Gambar/TRAINING/FZN'
GGZ_DIR='/home/pi/Documents/TA/Dataset_Gambar/TRAINING/GGZ'
LDY_DIR='/home/pi/Documents/TA/Dataset_Gambar/TRAINING/LDY'
NGH_DIR='/home/pi/Documents/TA/Dataset_Gambar/TRAINING/NGH'
NTA_DIR='/home/pi/Documents/TA/Dataset_Gambar/TRAINING/NTA'

#code to make data:
def assign_label(img,person_name):
    return person_name

X = []
y = []

def make_data(person_name,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,person_name)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100,100))
        
        X.append(np.array(img))
        y.append(str(label))

make_data(classes.get('ASING'), ASING_DIR)
make_data(classes.get('FAQ'), FAQ_DIR)
make_data(classes.get('FZN'), FZN_DIR)
make_data(classes.get('GGZ'), GGZ_DIR)
make_data(classes.get('LDY'), LDY_DIR)
make_data(classes.get('NGH'), NGH_DIR)
make_data(classes.get('NTA'), NTA_DIR)

#returns the shape (numberofImages,WIDTH,HEIGHT,CHANNELS)
len(X)
X = np.array(X)
X = X/255
print(X.shape)


