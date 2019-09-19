#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:20:31 2019

@author: Mani
"""

import pandas as pd
import numpy as np
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_val_score, learning_curve, validation_curve,cross_validate
from sklearn import tree, neural_network, ensemble, svm, neighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer

#from sklearn.metrics import accuracy_score

def readFile(y):
    if y == 0:
        data = pd.read_csv("heart.csv")
       
    else:
        data = pd.read_csv("Credit_Card.csv")
  
   
    return data 
def scaledData(x_data):
    sc = MinMaxScaler()
    x_data = sc.fit_transform(x_data)

    return x_data

    
def models():

     parameter_range = []
     parameter_name = []
 
    
     fnames = ["Heart.csv","Credit Card.csv"]
     mNames = ['Decision Tree', 'Neural Network', 'Boosting', 'SVM', 'KNN']
    
     for x in range(5):
        if x == 0:
            classificationModel = tree.DecisionTreeClassifier(max_depth=5)
        elif x == 1:
            classificationModel = neural_network.MLPClassifier()
        elif x == 2:
            classificationModel = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=5))
        elif x == 3:
            classificationModel = svm.SVC(gamma='scale', probability = True)
        else:
            classificationModel = neighbors.KNeighborsClassifier()
        
        for i in range(2):
            trainAccuracyResults =[]
            testAccuracyResults = []
            trainTimeResults =[]
            testTimeResults = []
            aucResults = []
            f1Results = []
            #crossValResults = []
            data = readFile(i)
       
            x_data = data.iloc[:,0:-1].values
            y_data = data.iloc[:,-1].values
 
            x_data = scaledData(x_data)
        
          
            trainSize, trainAccuracy, testAccuracy = learning_curve(classificationModel, x_data, y_data, train_sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], cv=5, shuffle=True, random_state=100)
            
    
            if x == 0:
                 parameter_range = [10,50,100,150,200]
                 parameter_name = 'max_depth'
            elif x == 1:
                 parameter_range=[(50,),(100,),(150,),(200,)]
                 parameter_name = 'hidden_layer_sizes'
            elif x == 2:
                parameter_range=[10,50,100,150,200]
                parameter_name = 'n_estimators'
            elif x == 3:
                parameter_range=[0.01, 0.1,1.0,10.0,100]
                parameter_name = 'gamma'
            elif x == 4:
                parameter_range=[1,10,20,30,40,50]
                parameter_name = 'n_neighbors'
                
            trainScores, testScores = validation_curve(classificationModel, x_data, y_data, param_name = parameter_name,param_range = parameter_range,cv=3)
            for z in range(len(trainSize)):
                x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = trainSize[z] , random_state =100, shuffle = True)
                
                start_time = timer()
                classificationModel.fit(x_train, y_train)
                train_time = timer() - start_time
                modelTrain = classificationModel.predict(x_train)
                start_time = timer()
                modelPredict = classificationModel.predict(x_test)
                #crossVal = cross_val_score(classificationModel, x_train, y_train, cv=3)
                test_time = timer() - start_time
                modelTrainAccuracy = mt.accuracy_score(y_train, modelTrain)
                modelTestAccuracy = mt.accuracy_score(y_test, modelPredict)
                
                trainAccuracyResults.append(modelTrainAccuracy)
                testAccuracyResults.append(modelTestAccuracy)
                #crossValResults.append(np.mean(crossVal))
                trainTimeResults.append(train_time)
                testTimeResults.append(test_time)
                probs = classificationModel.predict_proba(x_test)
                probs = probs[:,1]
                auc = mt.roc_auc_score(y_test, probs)
                aucResults.append(auc)
                f1 = mt.f1_score(y_test, modelPredict)
                f1Results.append(f1)

            plt.figure()
            plt.title(mNames[x] + '-> ' + fnames[i] + ', Learning Curves')
            plt.plot(trainSize,np.mean(trainAccuracy, axis=1),label='Training Accuracy - LC', linestyle='--')
           
 
            plt.plot(trainSize,np.mean(testAccuracy,axis=1),label='Test Accuracy - LC')
            plt.plot(trainSize, trainAccuracyResults, label = 'Training Accuracy',linestyle=':')
            plt.plot(trainSize, testAccuracyResults, label = 'Test Accuracy')
            #plt.plot(trainSize, crossValResults, label = 'CV Results')
            plt.xlabel('Training Size')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(mNames[x] + '-> ' + fnames[i] + ' Learning Curves' +'.png')
            
            plt.figure()
            plt.title(mNames[x] + '-> ' + fnames[i] + ', Validation Curves')
            plt.plot(parameter_range,np.mean(trainScores, axis=1),label='Train Scores')
 
            plt.plot(parameter_range,np.mean(testScores,axis=1),label='Test Scores')
            plt.xlabel(parameter_name)
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(mNames[x] + '-> ' + fnames[i] + ' Validation Curves' +'.png')
            
            plt.figure()
           
            
            
            plt.title(mNames[x] + '-> ' + fnames[i] + ', Time')
            plt.plot(trainSize,trainTimeResults,label='Training Time')
            plt.plot(trainSize,testTimeResults,label='Testing Time')
            plt.xlabel('Training Size')
            plt.ylabel('Seconds')
            plt.legend()
            plt.savefig(mNames[x] + '-> ' + fnames[i] + ' Time' +'.png')
            plt.figure()
            plt.title(mNames[x] + '-> ' + fnames[i] + ' ->, AOC and F1 Curves')
            plt.plot(trainSize,aucResults,label='AUC')
            plt.plot(trainSize,f1Results,label='F1')
            plt.xlabel('Training Size')
            plt.ylabel('AUC and F1')
            plt.legend()
            plt.savefig(mNames[x] + '-> ' + fnames[i] + ' AOC and F1 Curves' +'.png')
            
    
        plt.tight_layout()

def displayResults(df,x, y):
     fnames = ["Heart.csv","Credit Card.csv"]
     mNames = ['Decision Tree', 'Neural Network', 'Boosting', 'SVM', 'KNN']
   
     if x == 0:
         
         results1 = pd.pivot_table(df[df.param_criterion == 'gini'], values = 'mean_test_score', index = 'param_max_depth', columns = 'param_min_samples_split')
         results2 = pd.pivot_table(df[df.param_criterion == 'entropy'], values = 'mean_test_score', index = 'param_max_depth', columns = 'param_min_samples_split')
         plt.figure()
         ax5 = sns.heatmap(results1, cmap="YlGnBu")
         plt.title(mNames[x] + ' ' + fnames[y] + ', -> Gini')
         plt.savefig(mNames[x] + ' ' + fnames[y] + 'Gini' +'.png')
         plt.figure()
         ax6 = sns.heatmap(results2, cmap = 'Blues')
         plt.title(mNames[x] + ' ' + fnames[y] + ', -> Entropy')
         plt.savefig(mNames[x] + ' ' + fnames[y] + 'Entropy'+'.png')
         plt.tight_layout()
     elif x == 1:
         plt.figure()
         #results3 = pd.pivot_table(df, values = 'mean_test_score', index = 'param_solver', columns = 'param_max_iter')
         temp = df[df.param_solver == 'lbfgs']
         ax7 = sns.lineplot(temp.param_max_iter, temp.mean_test_score, label = 'lbfgs')
         temp = df[df.param_solver == 'sgd']
         ax7 = sns.lineplot(temp.param_max_iter, temp.mean_test_score, label = 'sgd')
         temp = df[df.param_solver == 'adam']
         ax7 = sns.lineplot(temp.param_max_iter, temp.mean_test_score, label = 'adam')
         plt.title(mNames[x] + ', -> ' + fnames[y])
         plt.savefig(mNames[x] + ' ' + fnames[y] +'.png')
     elif x == 2:
         plt.figure()
         results3 = pd.pivot_table(df, values = 'mean_test_score', index = 'param_n_estimators', columns = 'param_learning_rate')
         ax8 = sns.heatmap(results3, cmap ='Greens')
         plt.title(mNames[x] + ', -> ' + fnames[y])
         plt.savefig(mNames[x] + ' ' + fnames[y] +'.png')
     elif x == 3:
         plt.figure()
         temp = df[df.param_kernel == 'linear']
         ax9 = sns.lineplot(temp.param_C, temp.mean_test_score, label = 'linear')
         temp = df[df.param_kernel == 'rbf']
         ax10 = sns.lineplot(temp.param_C, temp.mean_test_score, label = 'rbf')
         plt.title(mNames[x] + ', -> ' + fnames[y])
         plt.savefig(mNames[x] + ' ' + fnames[y] +'.png')
     elif x == 4:
         plt.figure()
         temp = df[df.param_weights == 'uniform']
         ax11 = sns.lineplot(temp.param_n_neighbors, temp.mean_test_score, label = 'uniform')
         temp = df[df.param_weights == 'distance']
         ax12 = sns.lineplot(temp.param_n_neighbors, temp.mean_test_score, label = 'distance')
         plt.title(mNames[x] + ', -> ' + fnames[y])
         plt.savefig(mNames[x] + ' ' + fnames[y]+'.png')
         
         

         
        
def hyperTuning():
    mNames = ['Decision Tree', 'Neural Network', 'Boosting', 'SVM', 'KNN']
    fnames = ["Heart.csv","Credit Card.csv"]

    for x in range(5):
        
            if x == 0:
                classificationModel = tree.DecisionTreeClassifier()
                parameters = {'criterion':['gini','entropy'], 'max_depth':[10,20,30,40,50], 'min_samples_split':[2,4,6,8,10]}
            elif x == 1:
                classificationModel = neural_network.MLPClassifier()
                parameters = {'solver':['lbfgs','sgd','adam'], 'max_iter':[250,500,750]}
            elif x == 2:
                classificationModel = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=5))
                parameters = {'n_estimators':[75,100,125,150], 'learning_rate':[0.5, 0.75, 1.0, 1.25, 1.5]}
            elif x == 3:
                classificationModel = svm.SVC(gamma='scale')
                parameters = {'kernel':['linear','rbf'], 'C':[0.1,1,10,100]}
            else:
                 classificationModel = neighbors.KNeighborsClassifier()
                 parameters ={'n_neighbors':[1,3,5,7,9,11,13,15], 'weights':['uniform','distance']}
                 
            for y in range(2):
                data = readFile(y)
                x_data = data.iloc[:,0:-1].values
                y_data = data.iloc[:,-1].values
           
                x_data = scaledData(x_data)
                grid_search = GridSearchCV(classificationModel, param_grid=parameters, cv=3, n_jobs = -1)
                grid_search.fit(x_data, y_data)
             
               
                df = pd.DataFrame(grid_search.cv_results_)
                displayResults(df, x, y)
              
         
                print(mNames[x] + ' ' + fnames[y] + ' ' + 'Grid search best parameters', grid_search.best_params_)
                print(mNames[x] + ' ' + fnames[y] + ' ' + 'SVC Grid search best score', grid_search.best_score_)
             
                #print(mNames[x] + ' ' + fnames[y] + ' '+ 'SVC Grid search results', grid_search.cv_results_)
                
 
if __name__ == "__main__":
    models()
    hyperTuning()
   
  