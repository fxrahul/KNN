# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:08:28 2019

@author: 91755
"""
import pandas as pd
import numpy as np
import random,math,operator
from sklearn import preprocessing as pp
import os
from pathlib import Path
from datetime import datetime

continousFeature1 = []
continousFeature2 = []
categoricalFeature = []


data = pd.read_csv('data.csv') 

data.head()
inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
row, col = inputData.shape

#print(no_of_missing_values)

#Generating randomn index
def random_index(n):
    for i in range(n):
        index = random.sample(range(0,row),n)
    return index


def eucledian(x,y,length,colNo):
    d = 0
#    #type 1 scaling
#    if scalingOrNot == 1:
#        del x[colNo]
#        del y[colNo]
#        x = (pp.StandardScaler().fit(x)).transform(x)
#        y = (pp.StandardScaler().fit(y)).transform(y)
#        for i in range(0,length(x)):
#                d+= pow((x[i].astype(float) - y[i].astype(float)),2)
#    #type 2 scaling
#    elif scalingOrNot == 2:
#        del x[colNo]
#        del y[colNo]
#        x = (pp.StandardScaler().fit(x)).transform(x)
#        y = (pp.StandardScaler().fit(y)).transform(y)
#        for i in range(0,length(x)):
#                d+= pow((x[i].astype(float) - y[i].astype(float)),2)
#    else:
    for i in range(0,length):
            if i!=colNo :
                d+= pow((x[i].astype(float) - y[i].astype(float)),2)
#    print(math.sqrt(d))
    return math.sqrt(d)

def manhattan(x,y,length,colNo):
    d = 0
#    #type 1 scaling
#    if scalingOrNot == 1:
#        del x[colNo]
#        del y[colNo]
#        x = pp.scale(x)
#        y = pp.scale(y)
#        for i in range(0,length):
#            d+= abs(x[i].astype(float) - y[i].astype(float))
#     
#    #type 2 scaling
#    elif scalingOrNot == 2:
#        del x[colNo]
#        del y[colNo]
#        x = (pp.StandardScaler().fit(x)).transform(x)
#        y = (pp.StandardScaler().fit(y)).transform(y)
#        for i in range(0,length):
#            d+= abs(x[i].astype(float) - y[i].astype(float))
#    else:
    for i in range(0,length):
            if i!=colNo :
                d+= abs(x[i].astype(float) - y[i].astype(float))
    

    return d
        
    

def nn(training,testing,k,w,euc_or_man,colNo):
    if euc_or_man == 1:
        
        distance_with_training_set = []
        for row in training:
            distances = eucledian(testing,row,len(testing),colNo)
            distance_with_training_set.append((row,distances))
        
        
        distance_with_training_set.sort(key=operator.itemgetter(1))

        neigb = []
        if w==0 :
            for i in range(k):
                neigb.append(distance_with_training_set[i][0])
#            print(neigb)
            return neigb
        else :
            for i in range(k):
                neigb.append(distance_with_training_set[i])
    #        print(neigb)
    #        print("new")
    #        print(neigb[0][1])
            return neigb
    else :
        
        distance_with_training_set = []
        for row in training:
            distances = manhattan(testing,row,len(testing),colNo)
            distance_with_training_set.append((row,distances))
        
        
        distance_with_training_set.sort(key=operator.itemgetter(1))
        neigb = []
        if w==0 :
            for i in range(k):
                neigb.append(distance_with_training_set[i][0])
            return neigb
        else :
            for i in range(k):
                neigb.append(distance_with_training_set[i])
    #        print(neigb)
    #        print("new")
    #        print(neigb[0][1])
            return neigb
        
        
          
    
def predictValue(neigh,w,colNo):
    
    
    np.set_printoptions(suppress=True) #prevents exponential value
    a = []
    if(w==0) :

        
        for row in neigh[0]:
            a.append(row[colNo].astype(float))
        return round(np.mean(a))
    else :
        distances = []
        
        
        for row in neigh[0]:
            distances.append(row[1])
            a.append(row[0][colNo].astype(float))
#        print(distances)
        
#        weighted calculation
        lower = 0
        upper = 0
        for i in range(len(distances)):
            if distances[i] == 0:
                distances[i] = 1
            lower+= (1/distances[i])
            upper+= (1/distances[i]) * (a[i])
        
        
        return  (upper/lower)
#        return round(np.mean(a))
    
def predictValueClass(neigh,w):
    np.set_printoptions(suppress=True) #prevents exponential value
    a = []
    freq1 = 0
    freq2 = 0
    if w==0 :

        
        for row in neigh[0]:
            
            a.append(row[2].astype(float))
        
        
        for i in range(len(a)):
            if a[i] == 1.0:
                freq1+=1
            else :
                freq2+=1
        return 1.0 if freq1 > freq2 else 0.0
    else :
        distances = []
        
        
        for row in neigh[0]:
            distances.append(row[1])
            a.append(row[0][2].astype(float))
        for i in range(len(a)):
            if distances[i] == 0:
                distances[i] = 1
            if a[i] == 1.0:
                freq1+=(1/distances[i])
            else :
                freq2+=(1/distances[i])
        return 1.0 if freq1 > freq2 else 0.0

def predAccuracy(original,predValue):
    
    matched = 0
    for i in range(len(original)):
            if original[i] == predValue[i]:
                matched += 1
    accuracyVal = matched / float(len(original)) * 100.0       
    return accuracyVal

def getOriginalNeighbours(neighbours,wholeDatasetScaled , wholeDatasetOriginal):

    neigh = []
    neighbour = neighbours[0]
    
    wholeDatasetScaled = wholeDatasetScaled.astype(str)
    wholeDatasetOriginal = wholeDatasetOriginal.astype(str)
    for row in neighbour:
         index = np.where(np.all(wholeDatasetScaled==row  , axis=1 ))[0][0]
         result = wholeDatasetOriginal[index]
         neigh.append(result)
    r = [neigh]
    return r

def getOriginalNeighboursWeighted(neighbours,wholeDatasetScaled , wholeDatasetOriginal):

    neigh = []
    neighbour = neighbours[0]
    
    wholeDatasetScaled = wholeDatasetScaled.astype(str)
    wholeDatasetOriginal = wholeDatasetOriginal.astype(str)
    for row in neighbour:
         index = np.where(np.all(wholeDatasetScaled==row[0]  , axis=1 ))[0][0]
         result = wholeDatasetOriginal[index]
         result = (result,row[1])
         neigh.append(result)
    r = [neigh]
    
    return r
    
    


def fivePercentDataCol1():
    
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputData[:,0].astype(str) #Continous column
    column_2 = inputData[:,1].astype(str)  #Continous column
    column_3 = inputData[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 5) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_1[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,0].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
#    scalingPredicted_values = []
    
    
        #-----------------------prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
    
#     after row parameter, first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,,
#     third for e or m , fourrth for column no to omit for distance calculation
        neighbourse.append(nn(trainingSet, row, 1,0,1,0))
        neighbours.append(nn(trainingSet, row, 1,0,0,0))
#        scalingNeighboursEucledian.append(nn(trainingSet, row, 1,0,0,0))
        
    #first parameter neighbours, 2nd parameter for weighted , third parameter for column to omit
        predicted_valuese.append(predictValue(neighbourse,0,0).astype(int))
        predicted_values.append(predictValue(neighbours,0,0).astype(int))
#        scalingPredicted_values.append(predictValue(neighbours,0,0).astype(int))
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    print("accuracy for 1-nn on column1 for 5 percent missing value with eucledian distance",accuracye)
    continousFeature1.append(accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    print("accuracy for 1-nn on column1 for 5 percent missing value with manhattan distance",accuracy)
    continousFeature1.append(accuracy)
    
    
        #-------------------------- predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,0))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(neighbourse1,0,0).astype(int))
        predicted_values1.append(predictValue(neighbours1,0,0).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    print("accuracy for k-nn on column1 for 5 percent missing value with eucledian distance",accuracye1)
    continousFeature1.append(accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    print("accuracy for k-nn on column1 for 5 percent missing value with manhattan distance",accuracy1)
    continousFeature1.append(accuracy1)
#    
#        
#        
        #------------------------------predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,0))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(neighbourse2,1,0).astype(int))
        predicted_values2.append(predictValue(neighbours2,1,0).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    print("accuracy for weighted k-nn on column1 for 5 percent missing value with eucledian distance",accuracye2)
    continousFeature1.append(accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    print("accuracy for weighted k-nn on column1 for 5 percent missing value with manhattan distance",accuracy2)
    continousFeature1.append(accuracy2)
#---------------------------------------------------------------------------------------------------------
def fivePercentDataCol1Scale1():
    
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    inputDataScale = (pp.StandardScaler().fit(inputData)).transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 5) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_1[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,0].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
#    scalingPredicted_values = []
    
    
        #-----------------------prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
    
#     after row parameter, first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,,
#     third for e or m , fourrth for column no to omit for distance calculation
        neighbourse.append(nn(trainingSet, row, 1,0,1,0))
        neighbours.append(nn(trainingSet, row, 1,0,0,0))
#        scalingNeighboursEucledian.append(nn(trainingSet, row, 1,0,0,0))
        
    #first parameter neighbours, 2nd parameter for weighted , third parameter for column to omit
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,0).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,0).astype(int))
#        scalingPredicted_values.append(predictValue(neighbours,0,0).astype(int))
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature1.append(accuracye)
    print("after scaling type 1 accuracy for 1-nn on column1 for 5 percent missing value with eucledian distance ",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature1.append(accuracy)
    print("after scaling type 1 accuracy for 1-nn on column1 for 5 percent missing value with manhattan distance",accuracy)
    
    
        #-------------------------- predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,0))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,0).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,0).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature1.append(accuracye1)
    print("after scaling type 1 accuracy for k-nn on column1 for 5 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature1.append(accuracy1)
    print("after scaling type 1 accuracy for k-nn on column1 for 5 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
        #------------------------------predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,0))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,0).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,0).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature1.append(accuracye2)
    print("after scaling type 1 accuracy for weighted k-nn on column1 for 5 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature1.append(accuracy2)
    print("after scaling type 1 accuracy for weighted k-nn on column1 for 5 percent missing value with manhattan distance",accuracy2)


#---------------------------------------------------------------------------------------------------------
def fivePercentDataCol1Scale2():
    
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    inputDataScale = pp.MinMaxScaler().fit_transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 5) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_1[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,0].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
#    scalingPredicted_values = []
    
    
        #-----------------------prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
    
#     after row parameter, first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,,
#     third for e or m , fourrth for column no to omit for distance calculation
        neighbourse.append(nn(trainingSet, row, 1,0,1,0))
        neighbours.append(nn(trainingSet, row, 1,0,0,0))
#        scalingNeighboursEucledian.append(nn(trainingSet, row, 1,0,0,0))
        
    #first parameter neighbours, 2nd parameter for weighted , third parameter for column to omit
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,0).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,0).astype(int))
#        scalingPredicted_values.append(predictValue(neighbours,0,0).astype(int))
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature1.append(accuracye)
    print("after scaling type 2 accuracy for 1-nn on column1 for 5 percent missing value with eucledian distance ",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature1.append(accuracy)
    print("after scaling type 2 accuracy for 1-nn on column1 for 5 percent missing value with manhattan distance",accuracy)
    
    
        #-------------------------- predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,0))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,0).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,0).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature1.append(accuracye1)
    print("after scaling type 2 accuracy for k-nn on column1 for 5 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature1.append(accuracy1)
    print("after scaling type 2 accuracy for k-nn on column1 for 5 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
        #------------------------------predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,0))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,0).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,0).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature1.append(accuracye2)
    print("after scaling type 2 accuracy for weighted k-nn on column1 for 5 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature1.append(accuracy2)
    print("after scaling type 2 accuracy for weighted k-nn on column1 for 5 percent missing value with manhattan distance",accuracy2)



#---------------------------------------------------------------------------------------------------------
def tenPercentDataCol1Scale1():
    
    data = pd.read_csv('data.csv') 
    
    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    inputDataScale = (pp.StandardScaler().fit(inputData)).transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    row, col = inputData.shape
    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 10) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_1[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,0].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,0))
        neighbours.append(nn(trainingSet, row, 1,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,0).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,0).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature1.append(accuracye)
    print("after scale type 1 accuracy for 1-nn on column1 for 10 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature1.append(accuracy)
    print("after scale type 1 accuracy for 1-nn on column1 for 10 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,0))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,0).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,0).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature1.append(accuracye1)
    print("after scale type 1 accuracy for k-nn on column1 for 10 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature1.append(accuracy1)
    print("after scale type 1 accuracy for k-nn on column1 for 10 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,0))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,0).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,0).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature1.append(accuracye2)
    print("after scale type 1 accuracy for weighted k-nn on column1 for 10 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature1.append(accuracy2)
    print("after scale type 1 accuracy for weighted k-nn on column1 for 10 percent missing value with manhattan distance",accuracy2)

#------------------------------------------------------------------------------------------------------

def tenPercentDataCol1Scale2():
    
    data = pd.read_csv('data.csv') 
    
    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    inputDataScale = pp.MinMaxScaler().fit_transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    row, col = inputData.shape
    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 10) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_1[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,0].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,0))
        neighbours.append(nn(trainingSet, row, 1,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,0).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,0).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature1.append(accuracye)
    print("after scale type 2 accuracy for 1-nn on column1 for 10 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature1.append(accuracy)
    print("after scale type 2 accuracy for 1-nn on column1 for 10 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,0))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,0).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,0).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature1.append(accuracye1)
    print("after scale type 2 accuracy for k-nn on column1 for 10 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature1.append(accuracy1)
    print("after scale type 2 accuracy for k-nn on column1 for 10 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,0))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,0).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,0).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature1.append(accuracye2)
    print("after scale type 2 accuracy for weighted k-nn on column1 for 10 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature1.append(accuracy2)
    print("after scale type 2 accuracy for weighted k-nn on column1 for 10 percent missing value with manhattan distance",accuracy2)

#-------------------------------------------------------------------------------------------------------

def tenPercentDataCol1():
    
    data = pd.read_csv('data.csv') 
    
    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputData[:,0].astype(str) #Continous column
    column_2 = inputData[:,1].astype(str)  #Continous column
    column_3 = inputData[:,2].astype(str)  #CategoricalColumn
    
    row, col = inputData.shape
    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 10) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_1[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,0].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,0))
        neighbours.append(nn(trainingSet, row, 1,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(neighbourse,0,0).astype(int))
        predicted_values.append(predictValue(neighbours,0,0).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature1.append(accuracye)
    print("accuracy for 1-nn on column1 for 10 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature1.append(accuracy)
    print("accuracy for 1-nn on column1 for 10 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,0))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(neighbourse1,0,0).astype(int))
        predicted_values1.append(predictValue(neighbours1,0,0).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature1.append(accuracye1)
    print("accuracy for k-nn on column1 for 10 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature1.append(accuracy1)
    print("accuracy for k-nn on column1 for 10 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,0))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(neighbourse2,1,0).astype(int))
        predicted_values2.append(predictValue(neighbours2,1,0).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature1.append(accuracye2)
    print("accuracy for weighted k-nn on column1 for 10 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature1.append(accuracy2)
    print("accuracy for weighted k-nn on column1 for 10 percent missing value with manhattan distance",accuracy2)


#---------------------------------------------------------------------------------------------------------
def twentPercentDataCol1():
    data = pd.read_csv('data.csv') 
    
    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputData[:,0].astype(str) #Continous column
    column_2 = inputData[:,1].astype(str)  #Continous column
    column_3 = inputData[:,2].astype(str)  #CategoricalColumn
    
    row, col = inputData.shape
    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 20) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_1[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,0].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,0))
        neighbours.append(nn(trainingSet, row, 1,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(neighbourse,0,0).astype(int))
        predicted_values.append(predictValue(neighbours,0,0).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature1.append(accuracye)
    print("accuracy for 1-nn on column1 for 20 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature1.append(accuracy)
    print("accuracy for 1-nn on column1 for 20 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,0))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(neighbourse1,0,0).astype(int))
        predicted_values1.append(predictValue(neighbours1,0,0).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature1.append(accuracye1)
    print("accuracy for k-nn on column1 for 20 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature1.append(accuracy1)
    print("accuracy for k-nn on column1 for 20 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,0))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(neighbourse2,1,0).astype(int))
        predicted_values2.append(predictValue(neighbours2,1,0).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature1.append(accuracye2)
    print("accuracy for weighted k-nn on column1 for 20 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature1.append(accuracy2)
    print("accuracy for weighted k-nn on column1 for 20 percent missing value with manhattan distance",accuracy2)
#--------------------------------------------------------------------------------------------------------
    
def twentPercentDataCol1Scale1():
    data = pd.read_csv('data.csv') 
    
    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    inputDataScale = (pp.StandardScaler().fit(inputData)).transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    row, col = inputData.shape
    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 20) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_1[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,0].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,0))
        neighbours.append(nn(trainingSet, row, 1,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,0).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,0).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature1.append(accuracye)
    print("after scaling type 1 accuracy for 1-nn on column1 for 20 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature1.append(accuracy)
    print("after scaling type 1 accuracy for 1-nn on column1 for 20 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,0))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,0).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,0).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature1.append(accuracye1)
    print("after scaling type 1 accuracy for k-nn on column1 for 20 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature1.append(accuracy1)
    print("after scaling type 1 accuracy for k-nn on column1 for 20 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,0))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,0).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,0).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature1.append(accuracye2)
    print("after scaling type 1 accuracy for weighted k-nn on column1 for 20 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature1.append(accuracy2)
    print("after scaling type 1 accuracy for weighted k-nn on column1 for 20 percent missing value with manhattan distance",accuracy2)

#--------------------------------------------------------------------------------------------------------

def twentPercentDataCol1Scale2():
    data = pd.read_csv('data.csv') 
    
    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    inputDataScale = pp.MinMaxScaler().fit_transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    row, col = inputData.shape
    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 20) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_1[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,0].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,0))
        neighbours.append(nn(trainingSet, row, 1,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,0).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,0).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature1.append(accuracye)
    print("after scaling type 1 accuracy for 1-nn on column1 for 20 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature1.append(accuracy)
    print("after scaling type 1 accuracy for 1-nn on column1 for 20 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,0))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,0).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,0).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature1.append(accuracye1)
    print("after scaling type 1 accuracy for k-nn on column1 for 20 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature1.append(accuracy1)
    print("after scaling type 1 accuracy for k-nn on column1 for 20 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,0))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,0))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,0).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,0).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature1.append(accuracye2)
    print("after scaling type 1 accuracy for weighted k-nn on column1 for 20 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature1.append(accuracy2)
    print("after scaling type 1 accuracy for weighted k-nn on column1 for 20 percent missing value with manhattan distance",accuracy2)

    

#--------------------------------------------------------------------------------------------------------
def fivePercentDataCol2(): 
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputData[:,0].astype(str) #Continous column
    column_2 = inputData[:,1].astype(str)  #Continous column
    column_3 = inputData[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 5) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_2[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,1].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,1))
        neighbours.append(nn(trainingSet, row, 1,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(neighbourse,0,1).astype(int))
        predicted_values.append(predictValue(neighbours,0,1).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature2.append(accuracye)
    print("accuracy for 1-nn on column2 for 5 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature2.append(accuracy)
    print("accuracy for 1-nn on column2 for 5 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,1))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(neighbourse1,0,1).astype(int))
        predicted_values1.append(predictValue(neighbours1,0,1).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature2.append(accuracye1)
    print("accuracy for k-nn on column2 for 5 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature2.append(accuracy1)
    print("accuracy for k-nn on column2 for 5 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,1))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(neighbourse2,1,1).astype(int))
        predicted_values2.append(predictValue(neighbours2,1,1).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature2.append(accuracye2)
    print("accuracy for weighted k-nn on column2 for 5 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature2.append(accuracy2)
    print("accuracy for weighted k-nn on column2 for 5 percent missing value with manhattan distance",accuracy2)
#-----------------------------------------------------------------------------------------------------------

def fivePercentDataCol2Scale1(): 
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape

    inputDataScale = (pp.StandardScaler().fit(inputData)).transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 5) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_2[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,1].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,1))
        neighbours.append(nn(trainingSet, row, 1,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,1).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature2.append(accuracye)
    print("after scale type 1 accuracy for 1-nn on column2 for 5 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature2.append(accuracy)
    print("after scale type 1 accuracy for 1-nn on column2 for 5 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,1))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,1).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature2.append(accuracye1)
    print("after scale type 1 accuracy for k-nn on column2 for 5 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature2.append(accuracy1)
    print("after scale type 1 accuracy for k-nn on column2 for 5 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,1))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,1).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,1).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature2.append(accuracye2)
    print("after scale type 1 accuracy for weighted k-nn on column2 for 5 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature2.append(accuracy2)
    print("after scale type 1 accuracy for weighted k-nn on column2 for 5 percent missing value with manhattan distance",accuracy2)
#----------------------------------------------------------------------------------------------------------
    
def fivePercentDataCol2Scale2(): 
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape

    inputDataScale = pp.MinMaxScaler().fit_transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 5) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_2[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,1].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,1))
        neighbours.append(nn(trainingSet, row, 1,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,1).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature2.append(accuracye)
    print("after scale type 2 accuracy for 1-nn on column2 for 5 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature2.append(accuracy)
    print("after scale type 2 accuracy for 1-nn on column2 for 5 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,1))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,1).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature2.append(accuracye1)
    print("after scale type 2 accuracy for k-nn on column2 for 5 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature2.append(accuracy1)
    print("after scale type 2 accuracy for k-nn on column2 for 5 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,1))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,1).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,1).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature2.append(accuracye2)
    print("after scale type 2 accuracy for weighted k-nn on column2 for 5 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature2.append(accuracy2)
    print("after scale type 2 accuracy for weighted k-nn on column2 for 5 percent missing value with manhattan distance",accuracy2)


#-----------------------------------------------------------------------------------------------------------
def tenPercentDataCol2():
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputData[:,0].astype(str) #Continous column
    column_2 = inputData[:,1].astype(str)  #Continous column
    column_3 = inputData[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 10) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_2[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,1].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,1))
        neighbours.append(nn(trainingSet, row, 1,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(neighbourse,0,1).astype(int))
        predicted_values.append(predictValue(neighbours,0,1).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature2.append(accuracye)
    print("accuracy for 1-nn on column2 for 10 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature2.append(accuracy)
    print("accuracy for 1-nn on column2 for 10 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,1))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(neighbourse1,0,1).astype(int))
        predicted_values1.append(predictValue(neighbours1,0,1).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature2.append(accuracye1)
    print("accuracy for k-nn on column2 for 10 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature2.append(accuracy1)
    print("accuracy for k-nn on column2 for 10 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,1))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(neighbourse2,1,1).astype(int))
        predicted_values2.append(predictValue(neighbours2,1,1).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature2.append(accuracye2)
    print("accuracy for weighted k-nn on column2 for 10 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature2.append(accuracy2)
    print("accuracy for weighted k-nn on column2 for 10 percent missing value with manhattan distance",accuracy2)

#----------------------------------------------------------------------------------------------------------

def tenPercentDataCol2Scale1():
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    

    inputDataScale = (pp.StandardScaler().fit(inputData)).transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 10) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_2[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,1].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,1))
        neighbours.append(nn(trainingSet, row, 1,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,1).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature2.append(accuracye)
    print("after scale type 1 accuracy for 1-nn on column2 for 10 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature2.append(accuracy)
    print("after scale type 1 accuracy for 1-nn on column2 for 10 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,1))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,1).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature2.append(accuracye)
    print("after scale type 1 accuracy for k-nn on column2 for 10 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature2.append(accuracy1)
    print("after scale type 1 accuracy for k-nn on column2 for 10 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,1))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,1).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,1).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature2.append(accuracye2)
    print("after scale type 1 accuracy for weighted k-nn on column2 for 10 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature2.append(accuracy2)
    print("after scale type 1 accuracy for weighted k-nn on column2 for 10 percent missing value with manhattan distance",accuracy2)

  
#----------------------------------------------------------------------------------------------------------

def tenPercentDataCol2Scale2():
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    

    inputDataScale = pp.MinMaxScaler().fit_transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 10) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_2[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,1].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,1))
        neighbours.append(nn(trainingSet, row, 1,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,1).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature2.append(accuracye)
    print("after scale type 2 accuracy for 1-nn on column2 for 10 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature2.append(accuracy)
    print("after scale type 2 accuracy for 1-nn on column2 for 10 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,1))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,1).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature2.append(accuracye1)
    print("after scale type 2 accuracy for k-nn on column2 for 10 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature2.append(accuracy1)
    print("after scale type 2 accuracy for k-nn on column2 for 10 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,1))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,1).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,1).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature2.append(accuracye2)
    print("after scale type 2 accuracy for weighted k-nn on column2 for 10 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature2.append(accuracy2)
    print("after scale type 2 accuracy for weighted k-nn on column2 for 10 percent missing value with manhattan distance",accuracy2)

  
#----------------------------------------------------------------------------------------------------------
       
def twentPercentDataCol2():
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputData[:,0].astype(str) #Continous column
    column_2 = inputData[:,1].astype(str)  #Continous column
    column_3 = inputData[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 20) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_2[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,1].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,1))
        neighbours.append(nn(trainingSet, row, 1,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(neighbourse,0,1).astype(int))
        predicted_values.append(predictValue(neighbours,0,1).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature2.append(accuracye)
    print("accuracy for 1-nn on column2 for 20 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature2.append(accuracy)
    print("accuracy for 1-nn on column2 for 20 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,1))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(neighbourse1,0,1).astype(int))
        predicted_values1.append(predictValue(neighbours1,0,1).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature2.append(accuracye1)
    print("accuracy for k-nn on column2 for 20 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature2.append(accuracy1)
    print("accuracy for k-nn on column2 for 20 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,1))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(neighbourse2,1,1).astype(int))
        predicted_values2.append(predictValue(neighbours2,1,1).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature2.append(accuracye2)
    print("accuracy for weighted k-nn on column2 for 20 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature2.append(accuracy2)
    print("accuracy for weighted k-nn on column2 for 20 percent missing value with manhattan distance",accuracy2)

#---------------------------------------------------------------------------------------------------------

def twentPercentDataCol2Scale1():
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    inputDataScale = (pp.StandardScaler().fit(inputData)).transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 20) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_2[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,1].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,1))
        neighbours.append(nn(trainingSet, row, 1,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,1).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature2.append(accuracye)
    print("after scale type 1 accuracy for 1-nn on column2 for 20 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature2.append(accuracy)
    print("after scale type 1 accuracy for 1-nn on column2 for 20 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,1))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,1).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature2.append(accuracye1)
    print("after scale type 1 accuracy for k-nn on column2 for 20 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature2.append(accuracy1)
    print("after scale type 1 accuracy for k-nn on column2 for 20 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,1))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,1).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,1).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature2.append(accuracye2)
    print("after scale type 1 accuracy for weighted k-nn on column2 for 20 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature2.append(accuracy2)
    print("after scale type 1 accuracy for weighted k-nn on column2 for 20 percent missing value with manhattan distance",accuracy2)


#---------------------------------------------------------------------------------------------------------

def twentPercentDataCol2Scale2():
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    inputDataScale = (pp.StandardScaler().fit(inputData)).transform(inputData)
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputDataScale[:,0].astype(str) #Continous column
    column_2 = inputDataScale[:,1].astype(str)  #Continous column
    column_3 = inputDataScale[:,2].astype(str)  #CategoricalColumn
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 20) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_2[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,1].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    predicted_values = []
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        neighbours = []
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,1))
        neighbours.append(nn(trainingSet, row, 1,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese.append(predictValue(getOriginalNeighbours(neighbourse,inputDataScale,inputData),0,1).astype(int))
        predicted_values.append(predictValue(getOriginalNeighbours(neighbours,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    continousFeature2.append(accuracye)
    print("after scale type 1 accuracy for 1-nn on column2 for 20 percent missing value with eucledian distance",accuracye)
    accuracy = predAccuracy(original_value,predicted_values)
    continousFeature2.append(accuracy)
    print("after scale type 1 accuracy for 1-nn on column2 for 20 percent missing value with manhattan distance",accuracy)
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    predicted_values1 = []
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
        neighbours1 = []
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,1))
        neighbours1.append(nn(trainingSet, row1, 5,0,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese1.append(predictValue(getOriginalNeighbours(neighbourse1,inputDataScale,inputData),0,1).astype(int))
        predicted_values1.append(predictValue(getOriginalNeighbours(neighbours1,inputDataScale,inputData),0,1).astype(int))
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    continousFeature2.append(accuracye1)
    print("after scale type 1 accuracy for k-nn on column2 for 20 percent missing value with eucledian distance",accuracye1)
    accuracy1 = predAccuracy(original_value,predicted_values1)
    continousFeature2.append(accuracy1)
    print("after scale type 1 accuracy for k-nn on column2 for 20 percent missing value with manhattan distance",accuracy1)
#    
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    predicted_values2 = []
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        neighbours2 = []
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,1))
        neighbours2.append(nn(trainingSet, row2, 5,1,0,1))#first parameter is k value,second parameter for weighted : if 0 not weighted , else 1 ,, third for e or m , fourrth for column no to omit for distance calculation
    
    
        predicted_valuese2.append(predictValue(getOriginalNeighboursWeighted(neighbourse2,inputDataScale,inputData),1,1).astype(int))
        predicted_values2.append(predictValue(getOriginalNeighboursWeighted(neighbours2,inputDataScale,inputData),1,1).astype(int))
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    continousFeature2.append(accuracye2)
    print("after scale type 1 accuracy for weighted k-nn on column2 for 20 percent missing value with eucledian distance",accuracye2)
    accuracy2 = predAccuracy(original_value,predicted_values2)
    continousFeature2.append(accuracy2)
    print("after scale type 1 accuracy for weighted k-nn on column2 for 20 percent missing value with manhattan distance",accuracy2)


#---------------------------------------------------------------------------------------------------------
def fivePercentDataCol3():
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputData[:,0].astype(str) #Continous column
    column_2 = inputData[:,1].astype(str)  #Continous column
    column_3 = inputData[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 5) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_3[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,2].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []
    
    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
        
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,2))
        
    
    
        predicted_valuese.append((int) (predictValueClass(neighbourse,0)))

    

       ######################################
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    categoricalFeature.append(accuracye)
    categoricalFeature.append('NA')
    print("accuracy for 1-nn on column3 for 5 percent missing value with eucledian distance",accuracye)
   
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
    
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,2))
    
    
        predicted_valuese1.append((int) (predictValueClass(neighbourse1,0)))
     
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    categoricalFeature.append(accuracye1)
    categoricalFeature.append('NA')
    print("accuracy for k-nn on column3 for 5 percent missing value with eucledian distance",accuracye1)
  
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []

    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []

        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,2))
      
    
        predicted_valuese2.append( (int) (predictValueClass(neighbourse2,1) ) )
      
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    categoricalFeature.append(accuracye2)
    categoricalFeature.append('NA')
    for i in range(12):
        categoricalFeature.append('NA')
    print("accuracy for weighted k-nn on column3 for 5 percent missing value with eucledian distance",accuracye2)
   
    
def tenPercentDataCol3():
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputData[:,0].astype(str) #Continous column
    column_2 = inputData[:,1].astype(str)  #Continous column
    column_3 = inputData[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 10) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_3[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,2].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []

    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
    
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,2))
    
    
        predicted_valuese.append((int) (predictValueClass(neighbourse,0)))
       ######################################
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    categoricalFeature.append(accuracye)
    categoricalFeature.append('NA')
    print("accuracy for 1-nn on column3 for 10 percent missing value with eucledian distance",accuracye)
   
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    
    
    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []
    
        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,2))
    
    
        predicted_valuese1.append((int)(predictValueClass(neighbourse1,0)))
     
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    categoricalFeature.append(accuracye1)
    categoricalFeature.append('NA')
    print("accuracy for k-nn on column3 for 10 percent missing value with eucledian distance",accuracye1)
  
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []

    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
    
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,2))
      
    
        predicted_valuese2.append((int)(predictValueClass(neighbourse2,1)))
     
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    categoricalFeature.append(accuracye2)
    categoricalFeature.append('NA')
    for i in range(12):
        categoricalFeature.append('NA')
    print("accuracy for weighted k-nn on column3 for 10 percent missing value with eucledian distance",accuracye2)
   
        
def twentPercentDataCol3():
    data = pd.read_csv('data.csv') 

    data.head()
    inputData = pd.DataFrame(data, columns = ['a0', 'a1', 'a2']).to_numpy()
    row, col = inputData.shape
    

    #print(inputData)
    #print(np.amax(inputData))
    #first_col = inputData[:,0]/np.amax(inputData)
    #second_col = inputData[:,1]/np.amax(inputData)
    #third_col = inputData[:,2]
    #inputData = np.vstack((first_col,second_col,third_col)).T
    
    
    
    column_1 = inputData[:,0].astype(str) #Continous column
    column_2 = inputData[:,1].astype(str)  #Continous column
    column_3 = inputData[:,2].astype(str)  #CategoricalColumn
    
    
    
    

    #generating missing values according to index
    no_of_missing_values = (int) ( (row  * 20) /100 )
    index_col1 = random_index(no_of_missing_values)
    
    for i in range(len(index_col1)):
        column_3[index_col1[i]] = 'NA'
    
    new_data_5_percent = np.vstack((column_1,column_2,column_3)).T
    #print(new_data_5_percent)   
    #print(new_data_5_percent[index_col1[0]])
    testingSet = np.array([new_data_5_percent[index_col1[0]]])
    #print(testingSet)
    
    for i in range(1,len(index_col1)):
        testingSet=np.vstack([testingSet,new_data_5_percent[index_col1[i]]])
    
    trainingSet = np.delete(new_data_5_percent, index_col1, axis = 0 )
    original_value = [] #original column value
    col1_for_accuracy_measure = inputData[:,2].astype(float)
    
    for i in range(len(index_col1)):
        original_value.append(col1_for_accuracy_measure[index_col1[i]])
    
    
    
    
    predicted_valuese = []

    
    
    #prict value for each testing instances with nn
    for row in testingSet:
    
        neighbourse = []
    
        
        neighbourse.append(nn(trainingSet, row, 1,0,1,2))
    
    
        predicted_valuese.append((int)(predictValueClass(neighbourse,0)))
       ######################################
    
    
    accuracye = predAccuracy(original_value,predicted_valuese)
    categoricalFeature.append(accuracye)
    categoricalFeature.append('NA')
    print("accuracy for 1-nn on column3 for 20 percent missing value with eucledian distance",accuracye)
   
    
#    #predict value for for k-NN 
    predicted_valuese1 = []
    

    
    #prict value for each testing instances with nn
    for row1 in testingSet:
    
        neighbourse1 = []

        
        neighbourse1.append(nn(trainingSet, row1, 5,0,1,2))
    
    
        predicted_valuese1.append((int)(predictValueClass(neighbourse1,0)))
     
    
    
    accuracye1 = predAccuracy(original_value,predicted_valuese1)
    categoricalFeature.append(accuracye1)
    categoricalFeature.append('NA')
    print("accuracy for k-nn on column3 for 20 percent missing value with eucledian distance",accuracye1)
  
#        
#        
#    #predict value for weighted knn
#    
    predicted_valuese2 = []
    
    
    
    #prict value for each testing instances with nn
    for row2 in testingSet:
    
        neighbourse2 = []
        
        
        neighbourse2.append(nn(trainingSet, row2, 5,1,1,2))
      
    
        predicted_valuese2.append((int)(predictValueClass(neighbourse2,1)))
     
    
    
    accuracye2 = predAccuracy(original_value,predicted_valuese2)
    categoricalFeature.append(accuracye2)
    categoricalFeature.append('NA')
    for i in range(12):
        categoricalFeature.append('NA')
    print("accuracy for weighted k-nn on column3 for 20 percent missing value with eucledian distance",accuracye2)
   
    
   
if __name__ == "__main__":
    #for 5 percent missing values
    fivePercentDataCol1()
    fivePercentDataCol1Scale1()
    fivePercentDataCol1Scale2()

    #for 10 percent missing data
    
    tenPercentDataCol1()
    tenPercentDataCol1Scale1()
    tenPercentDataCol1Scale2()
    
    #for 20 percent missing values
    
    twentPercentDataCol1()
    twentPercentDataCol1Scale1()
    twentPercentDataCol1Scale2()
    
    print("-------------------------------------------")
    
    #for 5 percent missing values
    fivePercentDataCol2()
    fivePercentDataCol2Scale1()
    fivePercentDataCol2Scale2()

    #for 10 percent missing data
    
    tenPercentDataCol2()
    tenPercentDataCol2Scale1()
    tenPercentDataCol2Scale2()
    
    #for 20 percent missing values
    
    twentPercentDataCol2()
    twentPercentDataCol2Scale1()
    twentPercentDataCol2Scale2()
    
    print("-------------------------------------------")
    
    #for 5 percent missing values
    fivePercentDataCol3()


    #for 10 percent missing data
    
    tenPercentDataCol3()

    
    #for 20 percent missing values
    
    twentPercentDataCol3()
    
    

    column = ['Continous Feature 1','Continous Feature 2','Categorical Feature']
    indexs = ['5 percent Eucledian 1NN',
             '5 percent Mahattan 1NN',
             '5 percent Eucledian KNN',
             '5 percent Manhattan KNN',
             '5 percent Eucledian Weighted KNN',
             '5 percent Manhattan Weighted KNN',
             '5 percent Eucledian 1NN Scaling type 1',
             '5 percent Manhattan 1NN Scaling type 1',
             '5 percent Eucledian KNN Scaling type 1',
             '5 percent Manhattan KNN Scaling type 1',
             '5 percent Eucledian Weighted KNN Scaling type 1',
             '5 percent Manhattan Weighted KNN Scaling type 1',
             '5 percent Eucledian 1NN Scaling type 2',
             '5 percent Manhattan 1NN Scaling type 2',
             '5 percent Eucledian KNN Scaling type 2',
             '5 percent Manhattan KNN Scaling type 2',
             '5 percent Eucledian Weighted KNN Scaling type 2',
             '5 percent Manhataan Weighted KNN Scaling type 2',
             '10 percent Eucledian 1NN',
             '10 percent Mahattan 1NN',
             '10 percent Eucledian KNN',
             '10 percent Manhattan KNN',
             '10 percent Eucledian Weighted KNN',
             '10 percent Manhattan Weighted KNN',
             '10 percent Eucledian 1NN Scaling type 1',
             '10 percent Manhattan 1NN Scaling type 1',
             '10 percent Eucledian KNN Scaling type 1',
             '10 percent Manhattan KNN Scaling type 1',
             '10 percent Eucledian Weighted KNN Scaling type 1',
             '10 percent Manhattan Weighted KNN Scaling type 1',
             '10 percent Eucledian 1NN Scaling type 2',
             '10 percent Manhattan 1NN Scaling type 2',
             '10 percent Eucledian KNN Scaling type 2',
             '10 percent Manhattan KNN Scaling type 2',
             '10 percent Eucledian Weighted KNN Scaling type 2',
             '10 percent Manhataan Weighted KNN Scaling type 2',
             '20 percent Eucledian 1NN',
             '20 percent Mahattan 1NN',
             '20 percent Eucledian KNN',
             '20 percent Manhattan KNN',
             '20 percent Eucledian Weighted KNN',
             '20 percent Manhattan Weighted KNN',
             '20 percent Eucledian 1NN Scaling type 1',
             '20 percent Manhattan 1NN Scaling type 1',
             '20 percent Eucledian KNN Scaling type 1',
             '20 percent Manhattan KNN Scaling type 1',
             '20 percent Eucledian Weighted KNN Scaling type 1',
             '20 percent Manhattan Weighted KNN Scaling type 1',
             '20 percent Eucledian 1NN Scaling type 2',
             '20 percent Manhattan 1NN Scaling type 2',
             '20 percent Eucledian KNN Scaling type 2',
             '20 percent Manhattan KNN Scaling type 2',
             '20 percent Eucledian Weighted KNN Scaling type 2',
             '20 percent Manhataan Weighted KNN Scaling type 2'
             ]
    
    list_of_tuples = list(zip(continousFeature1,continousFeature2,categoricalFeature))
    df = pd.DataFrame(list_of_tuples, columns = column,index = indexs)

    dirName = 'result' +' '+str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
 
    try:
        # Create target Directory
        os.mkdir(dirName)
        
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
#    print(df)
    p = Path(dirName)
    df.to_csv(Path(p, 'results.csv'))
    
