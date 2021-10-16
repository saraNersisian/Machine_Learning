#-------------------------------------------------------------------------
# AUTHOR: Sara Nersisian
# FILENAME: decision_tree.py
# SPECIFICATION: training, testing and outputting the performance of a data set
# FOR: CS 4210- Assignment #2
# TIME SPENT: 9 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv


counter = 1   #counter for showing the specific training set

#function for transforming data to number representation for X
data = []          #random name for passing data from training and test sets
def transforming_data(data):
    X.clear()    #since the same function is used for both training and test data
                 #in order to not append the test data after training data in one array,
                 # X.clear() was used
    for row in range(len(data)):
        X_row = []
        for col in range(len(data[row]) - 1):
            if data[row][col] == "Young":
                X_row.append(1)
            elif data[row][col] == "Presbyopic":
                X_row.append(2)
            elif data[row][col] == "Prepresbyopic":
                X_row.append(3)
            elif data[row][col] == "Myope":
                X_row.append(1)
            elif data[row][col] == "Hypermetrope":
                X_row.append(2)
            elif data[row][col] == "Yes":
                X_row.append(1)
            elif data[row][col] == "No":
                X_row.append(2)
            elif data[row][col] == "Reduced":
                X_row.append(1)
            elif data[row][col] == "Normal":
                X_row.append(2)
        X.append(X_row)
    return X
    #print("X = ", X)

#function for transforming training class to numbers
dataClass = []
def transforming_class(dataClass):
    Y.clear()       #since the same function is used for both training and test classes
                    #in order to not append the test class after training class in one array,
                    # Y.clear() was used
    for row in range(len(dataClass)):
        if dataClass[row][4] == "Yes":
            Y.append(1)
        elif dataClass[row][4] == "No":
            Y.append(2)
    return Y
    #print("Y = " , Y)


dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append(row)
    print("Training set #" , counter , ": ", dbTraining)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # used A1 decision_tree code to find X and Y
    X = transforming_data(dbTraining)
    print("X = ", X)
    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    Y = transforming_class(dbTraining)
    print("Y = ", Y)

    #loop your training and test tasks 10 times here
    dbTest = []
    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)
       dbTest.clear()

       #read the test data and add this data to dbTest
       with open('contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for j, row in enumerate(reader):
               if j > 0:  # skipping the header
                   dbTest.append(row)
                 #  print(row)
       #print("dbTest = ", dbTest)

       TP = 0
       TN = 0
       FP = 0
       FN = 0
       row = 0      #variable for each row

       class_predicted = []

       for data in dbTest:

            X_test = transforming_data(dbTest)
            Y_test = transforming_class(dbTest)
            #print("X-test: ", X_test[row])
            #print("Y-test: ", Y_test[row])

           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label

            class_predicted.append(clf.predict([X_test[row]])[0])

           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.


            if (Y_test[row] == class_predicted[row]):
                if(Y_test[row] == 1):
                    TP = TP + 1
                elif(Y_test[row] == 2):
                    TN = TN + 1
            else:
                if (Y_test[row] == 1):
                    FN = FN + 1
                if (Y_test[row] == 2):
                    FP = FP + 1

            # print("\nFP=", FP)
            # print("FN=", FN)
            # print("TP=", TP)
            # print("TN=", TN)

            row = row + 1  # getting the next row, since X-test already converted the whole dbTest into numbers


            accuracy = (TP + TN) / (TP + TN + FN + FP)

       print("Accuracy = ", accuracy)

        #find the lowest accuracy of this model during the 10 runs (training and test set)

       class_predicted.clear()



    #print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here

    counter = counter + 1
    print('\n--------------------------------------- Next ----------------------------------------------\n')

