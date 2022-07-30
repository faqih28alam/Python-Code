#https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb#:~:text=The%20optimal%20K%20value%20usually,be%20aware%20of%20the%20outliers.
#https://medium.com/swlh/image-classification-with-k-nearest-neighbours-51b3a289280

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
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

##################preparing dataset
#set a variable to the dictionary of your class names:
classes = {'FAQIH':0,'FAUZAN':1,'GEDE':2,'LIDYA':3,'RIANTA':4, 'ASING':5}
label_prediksi = {0:'FAQIH',1:'FAUZAN',2:'GEDE',3:'LIDYA',4:'RIANTA',5:'ASING'}

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

make_data(classes.get('FAQIH'), FAQIH_DIR)
make_data(classes.get('FAUZAN'), FAUZAN_DIR)
make_data(classes.get('GEDE'), GEDE_DIR)
make_data(classes.get('LIDYA'), LIDYA_DIR)
make_data(classes.get('RIANTA'), RIANTA_DIR)
make_data(classes.get('ASING'), ASING_DIR)

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


#Finding Error Rate vs. K Value

# error_rate = []
# for i in range(1,40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train,y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.mean(pred_i != y_test))
# 
# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
#          marker='o',markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate))+1)
# plt.show()

#Finding accuracy vs. K Value

acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = knn.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc))+1)
plt.show()

# print("Using SKLEARN")
# lix = []
# liy = []
# index=0
# acc=0
# for k in range(1, 100):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#     liy.append(knn.score(X_test, y_test))
#     if liy[k-1]>acc:
#         acc=liy[k-1]
#         index=k-1
#     lix.append(k)
# 
# plt.plot(lix, liy)
# plt.show()
# print("max acc at k="+str(index+1)+" acc of "+str(acc))