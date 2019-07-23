import pandas as pd
import cv2
import os
import time
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import pyspark


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
import mahotas
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("TkAgg")
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

starttime = time.time()
path = os.getcwd()
classes = ['airplane','car','cat','dog','flower','fruit','motorbike','person']
image_Details = {}
bins = 8


# Reading the images to collect features
def final(path,class_Val,target_Val):
    os.listdir(path + '/' +classes[class_Val])
    image_Name = sorted(os.listdir(path + '/' + classes[class_Val]))
    os.chdir(path + '/' + classes[class_Val])
    for img in image_Name:
        if img != '.DS_Store':
            class_List = [target_Val]
            image = cv2.imread(img)
            gray_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f1 = cv2.resize(gray_Image, (100, 100)).flatten().tolist()
            feature = cv2.HuMoments(cv2.moments(gray_Image)).flatten().tolist()
            haralick = mahotas.features.haralick(gray_Image).mean(axis=0).flatten().tolist()
            hist_Image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist  = cv2.calcHist([hist_Image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            hist_Feature = hist.flatten().flatten().tolist()
            global_Feature = class_List + f1 + feature + haralick + hist_Feature
            image_Details[img] = global_Feature
        else:
            continue
    return image_Details

image_Airplane = final(path,0,1)
image_Car = final(path,1,2)
image_Cat = final(path,2,3)
image_Dog = final(path,3,4)
image_Flower = final(path,4,5)
image_Fruit = final(path,5,6)
image_Bike = final(path,6,7)
image_Person = final(path,7,8)

#A dictionary of all the images and features
image_All = image_Airplane

image_All_df = pd.DataFrame.from_dict(image_All,orient='index')

X = image_All_df.iloc[:,1:len(image_All_df.columns)]
X = X.values
Y = image_All_df.iloc[:,0]
Y = Y.values

#Preapring the training and testing data
X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)

# ADA Boost with SVM
clf = AdaBoostClassifier(SVC(probability=True, kernel='linear'))
clf.fit(X_train,Y_train)
clf.predict(X_train)
Y_pred = clf.predict(X_test)
print("accuracy is:",accuracy_score(Y_test,Y_pred))

print("Seconds:%s" % (time.time() - starttime))

#SVM
scaler = StandardScaler()

scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y_train)


clf = SVC().fit(X_train,Y_train)
clf.predict(X_train)
Y_pred = clf.predict(X_test)
print("SVM Accuracy: {}%".format(clf.score(X_test, Y_test) * 100 ))
print("Seconds:%s" % (time.time() - starttime))

#ADA Boost with Decision Tree
from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm="SAMME",n_estimators=200, learning_rate=0.8)
ada.fit(X_train,Y_train)
Y_pred=ada.predict(X_test)
print("RMSE of ADA boost with decision tree is:",metrics.mean_squared_error(Y_test, Y_pred))
print("Seconds:%s" % (time.time() - starttime))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_test)
print("Logistic Regression RMSE is:",metrics.mean_squared_error(Y_test, y_pred))
t = time.time() - starttime
print("Seconds:%s" % (t*6))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rdf=RandomForestClassifier()
rdf.fit(X_train,Y_train)
Y_pred=rdf.predict(X_test)
print("Random Forest Accuracy is:",metrics.accuracy_score(Y_test, Y_pred))
t = (time.time() - starttime)
print("Time taken to execute Random Forest:%s" % (t) )
