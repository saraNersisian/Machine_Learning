#-------------------------------------------------------------------------
# AUTHOR: Sara Nersisian
# FILENAME: find_s.py
# SPECIFICATION: find_s program for machine learning
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
import csv

num_attributes = 4
db = []
print("\n The Given Training Data Set \n")

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes #representing the most specific possible hypothesis
print(hypothesis)

#find the first positive training data in db and assign it to the vector hypothesis
#Selecting only the positive
for row in db:
        if row[4] == 'Yes':
            hypothesis = row
            break
#updating hypothesis to have 4 main features only
hypothesis = hypothesis[0:4]
# print("h1:", hypothesis)


#find the maximally specific hypothesis according to your training data in db and assign it to the vector hypothesis (special characters allowed: "0" and "?")

#creating an array to only hold the postive attributes
dbYes=[]
for row in db:
    if row[4] == "Yes":
        dbYes.append(row[:4])
# print(dbYes)

# comparing each value of dbYes to hypothesis
# if they are not equal print "?"

i = 0               #index for printing steps of hypothesis
for row in dbYes:
    for col in row:
        if col != hypothesis[row.index(col)]:
            hypothesis[row.index(col)] = "?"
    i+=1
    print("h",i,":", hypothesis)



print("\n The Maximally Specific Hypothesis for the given training examples found by Find-S algorithm:\n")
print(hypothesis)
