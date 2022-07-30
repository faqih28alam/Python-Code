#https://www.youtube.com/watch?v=TKfTKA5EmmY&t=130s
#https://stackoverflow.com/questions/70945386/how-to-load-custom-image-dataset-to-x-train
#https://vitalflux.com/python-draw-confusion-matrix-matplotlib/

##################importing libraries
import sklearn
from tqdm import tqdm
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt



##################preparing dataset
#set a variable to the dictionary of your class names:
classes = {'FAQIH':0,'FAUZAN':1,'GEDE':2,'LIDYA':3,'RIANTA':4, 'ASING':5}

FAQIH_DIR  = '/home/pi/Documents/FAQIH/Dataset/TRAIN/FAQIH'
FAUZAN_DIR = '/home/pi/Documents/FAQIH/Dataset/TRAIN/FAUZAN'
GEDE_DIR   = '/home/pi/Documents/FAQIH/Dataset/TRAIN/GEDE'
LIDYA_DIR  = '/home/pi/Documents/FAQIH/Dataset/TRAIN/LIDYA'
RIANTA_DIR = '/home/pi/Documents/FAQIH/Dataset/TRAIN/RIANTA'
ASING_DIR  = '/home/pi/Documents/FAQIH/Dataset/TRAIN/ASING'

#code to make data:
def assign_label(img,person_name):
    return person_name

X = []
y = []
Z = []

def make_data(person_name,DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img,person_name)
        path  = os.path.join(DIR,img)
        img   = cv2.imread(path)
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img   = cv2.resize(img, (100,100)).flatten()
        
        X.append(np.array(img))
        y.append(str(label))

make_data(classes.get('ASING'), ASING_DIR)
make_data(classes.get('FAQIH'), FAQIH_DIR)
make_data(classes.get('FAUZAN'), FAUZAN_DIR)
make_data(classes.get('GEDE'), GEDE_DIR)
make_data(classes.get('LIDYA'), LIDYA_DIR)
make_data(classes.get('RIANTA'), RIANTA_DIR)

#returns the shape (numberofImages,WIDTH,HEIGHT,CHANNELS)
print("jumlah data pada variabel X : ", len(X))
X = np.array(X)
X = X/255
print("shape variabel X : ", X.shape)
y = np.array(y)
print("shape variabel y : ", y.shape)

print("shape data gambar ke 1", X[0].shape)
print("list array data gambar data ke 1", X[0])

print("shape data label ke 449", y[450].shape)
print("list numpy array data label ke 449", y[450])

print("--------------------------------------------------------")
 
###################Split data
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99, stratify = y)
 
print("Shape X_train ", X_train.shape)
print("Shape X_test ",X_test.shape)
print("Shape y_train",y_train.shape)
print("Shape y_test",y_test.shape)

print("--------------------------------------------------------")

####################Fit the model
model = KNeighborsClassifier(n_neighbors = 9)
model.fit(X_train, y_train)

#################### save the model
pickle.dump(model, open("ModelKNN.K9.pkl", "wb"))

####################Make prediction
y_pred = model.predict(X_test)
print(y_pred)

print("--------------------------------------------------------")

####################Evaluate the model
report = classification_report(y_test,y_pred)
print(report)

print("--------------------------------------------------------")

confusion = confusion_matrix(y_true = y_test, y_pred = y_pred)
print(confusion)
print("--------------------------------------------------------")

####################Test Akurasi Model dan Matrix
acc = model.score(X_test, y_test)
print(f"accuracy from knn.score = {acc:.4}")

print("--------------------------------------------------------")

acc = metrics.accuracy_score(y_test, y_pred)
print(f"accuracy from metrics.accuracy_score = {acc:.4}")

print("--------------------------------------------------------")

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(x=j, y=i,s=confusion[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix KNN of 6 classes', fontsize=18)
plt.show()

print("--------------------------------------------------------")