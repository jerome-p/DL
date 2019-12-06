##################################################
## 1. Load breast cancer dataset from sklearn
## 2. Split the data as 70:30 as train and test data
## 3. Fit the train data into SVM model with diffferent kernels
##    and bar plot the accuracy of different SVM model with the test data
## 4. Fit the above training dataset into a SVM model with ploynomial kernel
##    with varying degree and plot the accuracy wrt. degree of ploynomial kernel with the test data
## 5. Define a custom kernel K(X,Y)=K*XY'+theta where k and theta are constants
## 6. Use the custom kernel and report the accuracy with the given train and test dataset
##################################################

##################################################
## Basic imports
## You are not required to import additional module imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import seaborn as sns
import pandas as pd
###################################################

###################################################
## the method loads breast cancer dataset and returns
## the dataset and label as X,Y
def load_data(): 
	data_set = datasets.load_breast_cancer()
	X=data_set.data
	y=data_set.target
	return X,y
###################################################

###################################################
## this method takes train and test data and different 
## svm models and fit the train data into svm models and 
## do bar plot using sns.barplot() of different svm model 
## accuracy. You need to implement the model fitting and 
## bar plotting in this method.
def svm_models(X_train, X_test, y_train, y_test,models):
	## write your own code here
    model=['linear','rbf','poly']
    l=len(models)
    accuracy=[]
    i=l-1
    while i>=0:
        c=models[i]
        c.fit(X_train,y_train)
        y_pred=c.predict(X_test)
        acc=metrics.accuracy_score(y_test, y_pred)
        accuracy.append(acc)
        i=i-1
    print(accuracy)
    svm_accuracy=pd.DataFrame(columns=['models','accuracy'])
    svm_accuracy['models']=model
    svm_accuracy['accuracy']=accuracy
    sns.set(style="whitegrid")
    sns.barplot(x="models", y="accuracy", data=svm_accuracy)
    plt.show()
    ###################################################

###################################################
## this method fits the dataset to a svm model with 
## polynomial kernel with degree varies from 1 to 3 
## and plots the execution time wrt. degree of 
## polynomial, you can calculate the elapsed time 
## by time.time() method
def ploy_kernel_var_deg(X_train, X_test, y_train, y_test):
	## write your own code here
    models = (svm.SVC(kernel='poly', degree=1),svm.SVC(kernel='poly', degree=2),
           svm.SVC(kernel='poly', degree=3))
    degree=['1','2','3']
    l=len(models)
    accuracy=[]
    i=l-1
    while i>=0:
        c=models[i]
        c.fit(X_train,y_train)
        y_pred=c.predict(X_test)
        acc=metrics.accuracy_score(y_test, y_pred)
        accuracy.append(acc)
        i=i-1
    print(accuracy)
    svm_accuracy=pd.DataFrame(columns=['degree','accuracy'])
    svm_accuracy['degrees']=degree
    svm_accuracy['accuracy']=accuracy
    sns.set(style="whitegrid")
    sns.barplot(x="models", y="accuracy", data=svm_accuracy)
    plt.show()
###################################################

###################################################
## this method implements a custom kernel technique 
## which is K(X,Y)=k*XY'+theta where k and theta are
## constants. Since SVC supports custom kernel function
## with only 2 parameters we return the custom kernel 
## function name from another method which takes k and
## theta as input
def custom_kernel(k,theta):
    
    def my_kernel(X, Y):
	## write your own code here
        r=np.dot(X,Y.T)*k+theta
        return r
    	## write your own code here
    return my_kernel

####################################################

####################################################
## this method uses the custom kernel and fit the 
## training data and reports accuracy on test data
def svm_custom_kernel(X_train, X_test, y_train, y_test, model):
	## write your code here
    pass
####################################################

####################################################
## main method:
def main():
	X,y=load_data()
	# Split dataset into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109) # 70% training and 30% test
	
	C=1
	models = (svm.SVC(kernel='linear', C=C),
          	svm.SVC(kernel='rbf', gamma='auto', C=C),
          	svm.SVC(kernel='poly', degree=2, gamma='auto', C=C))

	svm_models(X_train, X_test, y_train, y_test,models)
	
	ploy_kernel_var_deg(X_train, X_test, y_train, y_test)
	
	k=0.1
	theta=0.1
    
	svm_custom_kernel(X_train, X_test, y_train, y_test, model=svm.SVC(kernel=custom_kernel(k,theta)))
#####################################################	


if __name__=='__main__':
	main()



	
