#-------------------------------------------------------------------------
# AUTHOR: Sara Nersisian
# FILENAME: svm.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: start 10:30
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv

dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1])

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

for C in c: #iterates over c
    for D in degree: #iterates over degree
        for K in kernel: #iterates kernel
           for FSH in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C, D, K, FSH)


                #Fit SVM to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                correctGuess = 0
                maxAccuracy = 0
                for i, testSample in enumerate(dbTest):
                    class_predicted = int(clf.predict([testSample[:-1]]))

                    correctGuess += 1 * (class_predicted == int(testSample[-1]))

                accuracy = correctGuess / len(dbTest)
                if accuracy > maxAccuracy :
                    maxAccuracy = accuracy


                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                print("Highest SVM accuracy do far: ",maxAccuracy, ", Parameters: c=", C, " degree=", D, " kernel=",K ," decision_function_shape=",  FSH )











