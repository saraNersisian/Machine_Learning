#-------------------------------------------------------------------------
# AUTHOR: Sara Nersisian
# FILENAME: KNN.py
# SPECIFICATION: finding 1NN
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
class_numbers = {"+": 0, "-": 1}

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

print(db)
#loop your data to allow each instance to be your test set
wrong = 0       #counter for number of incorrect predictions

for i, instance in enumerate(db):
    X = []

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    for j in range(len(db)):
        if i != j:
            newArr = []
            for k in range(len(db[j]) - 1):         #extracting x and y, also excluding the test set from the rest of the data
                newArr.append(float(db[j][k]))
            X.append(newArr)

    #print(X)
    #X =

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    Y = []
    for j in range(len(db)):
        if i != j:      #excluding class of the test set from the training set
            Y.append(class_numbers[db[j][2]])
    #print(Y)
    #store the test sample of this iteration in the vector testSample
    testSample = []
    for j in range(len(instance) - 1):
        testSample.append(float(instance[j]))


    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]

    class_predicted = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != class_numbers[instance[len(instance) - 1]]:
        wrong += 1
#print the error rate
#--> add your Python code here
errorRate = wrong / len(db)      #incorrect/total
print(" ===> Error rate : " + str(errorRate))





