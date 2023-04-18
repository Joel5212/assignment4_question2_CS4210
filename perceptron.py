#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
import math
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

neural_network_classifier = ''
highest_preceptron_accuracy = -math.inf
highest_mlp_accuracy = -math.inf

for n_value in n: 
    
    #iterates over n

    for r_value in r: #iterates over r

        #iterates over both algorithms
        #-->add your Pyhton code here
        for a in range(2): #iterates over the algorithms
            
            
            #Create a Neural Network classifier
            #if Perceptron then
            if a==0:
            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
                clf = Perceptron(eta0=n_value, shuffle=r_value, max_iter=1000)
                clsf= "Highest Preceptron"
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer,
            #                          shuffle = shuffle the training data, max_iter=1000
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=n_value, hidden_layer_sizes=(25,), shuffle = r_value, max_iter=1000) 
                clsf = "Highest Mulit Layer Perceptron"
            #-->add your Pyhton code here

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here

            correct_amount = 0


            for(x_testSample, y_testSample) in zip(X_test, y_test):
                predict = clf.predict([x_testSample])[0]
                if predict == y_testSample:
                    correct_amount += 1
            
            current_accuracy = correct_amount / len(y_test)

            if a == 0:
                if current_accuracy > highest_preceptron_accuracy:
                    highest_perceptron_accuracy = current_accuracy
                    print(clsf + " acccuracy so far: " + str(highest_perceptron_accuracy) + "\n Parameters:" + " learning rate = " + str(n_value) + " shuffle = " + str(r_value))
                    print("\n")
            else:
                if current_accuracy > highest_mlp_accuracy:
                    highest_mlp_accuracy = current_accuracy
                    print(clsf + " acccuracy so far: " + str(highest_mlp_accuracy) + "\n Parameters:" + " learning rate = " + str(n_value) + " shuffle = " + str(r_value))     
                    print("\n")

print("\nHighest Perceptron Accuracy: " + str(highest_perceptron_accuracy))
print("Highest Multi Layer Perceptron: " + str(highest_mlp_accuracy))
            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code h









