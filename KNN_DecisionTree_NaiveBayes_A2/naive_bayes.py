#-------------------------------------------------------------------------
# AUTHOR: Sara Nersisian
# FILENAME: naive_bayes.py
# SPECIFICATION: classifying a new instance based on Naive bayes strategy
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv


data = []          #random name for passing data from training and test sets
X = []
def transforming_data(data):
    X.clear()    #since the same function is used for both training and test data
                 #in order to not append the test data after training data in one array,
                 # X.clear() was used
    for r in range(len(data)):
        X_row = []
        for col in range(len(data[r])- 1):
            if data[r][col] == "Sunny":
                X_row.append(1)
            elif data[r][col] == "Overcast":
                X_row.append(2)
            elif data[r][col] == "Rain":
                X_row.append(3)
            elif data[r][col] == "Hot":
                X_row.append(1)
            elif data[r][col] == "Mild":
                X_row.append(2)
            elif data[r][col] == "Cool":
                X_row.append(3)
            elif data[r][col] == "High":
                X_row.append(1)
            elif data[r][col] == "Normal":
                X_row.append(2)
            elif data[r][col] == "Weak":
                X_row.append(1)
            elif data[r][col] == "Strong":
                X_row.append(2)
            elif data[r][col] == "Yes":
                X_row.append(1)
            elif data[r][col] == "No":
                X_row.append(2)
        X.append(X_row)
    return X

dataClass = []
Y = []
def transforming_class(dataClass):
    Y.clear()       #since the same function is used for both training and test classes
                    #in order to not append the test class after training class in one array,
                    # Y.clear() was used
    for row in range(len(dataClass)):
        if dataClass[row][5] == "Yes":
            Y.append(1)
        elif dataClass[row][5] == "No":
            Y.append(2)
    return Y
    #print("Y = " , Y)


db = []
#reading the training data
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

print(db)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
X = transforming_data(db)
print(X)

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = transforming_class(db)
print(Y)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
dbTest = []
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTest.append (row)

print(dbTest)
X_test = transforming_data(dbTest)
print(X_test)
#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
## this part is not complete !!!!!!!!!!!!
predicted = clf.predict_proba(X_test)[0]
print(predicted)


