#This code generates random data from Uniform Distribution and assigns labels.
#The data is a non-linear one with points inside a circle of fixed radius marked as -1 and outside as +1.
#We flip the labels of some data (here with 5% probability) to introduce some noise.
#You will be using Decision Tree and Naive Bayes Classifiers to classify the above generated data.


import numpy as np
import matplotlib.pyplot as plt
#Do all the necessary imports here

def generate_data():

	np.random.seed(123) #Set seed for reproducibility. Please do not change/remove this line.
	x = np.random.uniform(-1,1,(128,2)) #You may change the number of samples you wish to generate
	y=[]
	for i in range(x.shape[0]):
		y.append(np.sign(x[i][0]**2 + x[i][1]**2 - 0.5)) #Forming labels
	return x,y

def flip_labels(y):

	num = int(0.05 * len(y)) #5% of data to be flipped
	np.random.seed(123)
	changeind = np.random.choice(len(y),num,replace=False) #Sampling without replacement
	#For example, np.random.choice(5,3) = ([0,2,3]); first argument is the limit till which we intend to pick up elements, second is the number of elements to be sampled

	#Creating a copy of the array to modify
	yc=np.copy(y) # yc=y is a bad practice since it points to the same location and changing y or yc would change the other which won't be desired always
	#Flip labels -1 --> 1 and 1 --> -1
	for i in changeind:
		if yc[i]==-1.0:
			yc[i]=1.0
		else:
			yc[i]=-1.0

	return yc

#Fill up the below function
def train_test_dt(x,y):

	# Perform a k-fold cross validation using Decision Tree
	# Plot train and test accuracy with varying k (1<=k<=10)
	dt = DecisionTreeClassifier(random_state=0)
    train_accuracy = []
    test_accuracy = []
    for i in range(2,11):
        eval_vals = cross_validate(dt, x, y, cv=i, return_train_score=True)
        mean_train_acc = eval_vals['train_score'].mean()
        mean_test_acc = eval_vals['test_score'].mean()
        train_accuracy.append(mean_train_acc)
        test_accuracy.append(mean_test_acc)
    k = [i for  i in range(2,11)]
    plt.plot(k,train_accuracy,'g-',label='Train Accuracy')
    plt.plot(k,test_accuracy,'c-',label='Test Accuracy')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('K',fontsize=15)
    plt.ylabel('Average Accuracy',fontsize=15)
    plt.title('Test Train accuracies',fontsize=15)
    plt.legend(fontsize=20)
    plt.show()

#Fill up the velow function
def train_test_nb(x,y):

	# Perform a k-fold cross validation using Decision Tree
    # Plot train and test accuracy with varying k (1<=k<=10)

    nb = naive_bayes.GaussianNB()
    train_accuracy = []
    test_accuracy = []
    for i in range(2,11):
        eval_vals = cross_validate(nb, x, y, cv=i, return_train_score=True)
        mean_train_acc = eval_vals['train_score'].mean()
        mean_test_acc = eval_vals['test_score'].mean()
        train_accuracy.append(mean_train_acc)
        test_accuracy.append(mean_test_acc)
    k = [i for  i in range(2,11)]
    plt.plot(k,train_accuracy,'g-',label='Train Accuracy')
    plt.plot(k,test_accuracy,'c-',label='Test Accuracy')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('K',fontsize=15)
    plt.ylabel('Average Accuracy',fontsize=15)
    plt.title('Test Train accuracies',fontsize=15)
    plt.legend(fontsize=20)
    plt.show()



def main():

	x,y = generate_data() #Generate data
	y = flip_labels(y) #Flip labels
	y=np.asarray(y) #Change list to array
	train_test_dt(x,y)
	train_test_nb(x,y)


if __name__=='__main__':
	main()
