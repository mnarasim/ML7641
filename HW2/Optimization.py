#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:34:23 2019

@author: Mani

The following code uses optimization algorithms built by Genevieve Hayes as below: https://github.com/gkhayes/mlrose
"""

import mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

    
def readFile():
    data = pd.read_csv("Heart.csv")

    return data 


def scaledData(x_data):
    sc = MinMaxScaler()
    x_data = sc.fit_transform(x_data)

    return x_data

def nnModelFunction(iMin, iMax, iStep, jMin, jMax, jStep, sMin, sMax, sStep):
    nnOptAlgos = ['random_hill_climb','simulated_annealing','genetic_alg']
    data = readFile()
    x_data = data.iloc[:,0:-1].values
    y_data = data.iloc[:,-1].values
    x_data = scaledData(x_data)
    optResults = {'algorithm':[],'training_size':[],'training_accuracy':[],'test_accuracy':[],'cv_training_accuracy':[], 'cv_test_accuracy':[]}
    
    for y in range(len(nnOptAlgos)):
        for x in np.arange(sMin,sMax,sStep):
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = x, random_state =100, shuffle = True)
           
            nnModel = mlrose.NeuralNetwork(hidden_nodes =[25],activation='sigmoid', algorithm=nnOptAlgos[y], max_iters = 50, max_attempts=5,learning_rate = 0.001,early_stopping=True, random_state =10)
            
            nnModel.fit(x_train, y_train)
            y_train_pred = nnModel.predict(x_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
            y_test_pred = nnModel.predict(x_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            optResults['training_size'].append(len(x_train))
                   
            optResults['training_accuracy'].append(y_train_accuracy)
            optResults['test_accuracy'].append(y_test_accuracy)
            cv_results = cross_validate(nnModel,x_train, y_train, cv= 5)
            optResults['cv_training_accuracy'].append(np.mean(cv_results['train_score']))
            optResults['cv_test_accuracy'].append(np.mean(cv_results['test_score']))
            optResults['algorithm'].append(nnOptAlgos[y])
           
    optResults = pd.DataFrame(optResults) 
    nnGraphs(optResults, nnOptAlgos)
    maxTrainingSize = optResults.loc[optResults.cv_test_accuracy.idxmax(),'training_size']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = maxTrainingSize, random_state =100, shuffle = True)
    optTuning = {'algorithm':[],'training_size':[],'training_accuracy':[],'test_accuracy':[],'cv_training_accuracy':[], 'cv_test_accuracy':[], 'iterations':[],'attempts':[]}
    for y in range(len(nnOptAlgos)):    
        for i in range(iMin, iMax, iStep):
            for j in range(jMin, jMax, jStep):
           
                nnModel = mlrose.NeuralNetwork(hidden_nodes =[25],activation='sigmoid', algorithm=nnOptAlgos[y], max_iters = i, max_attempts=j,learning_rate = 0.001,early_stopping=True, random_state =10)
                
                nnModel.fit(x_train, y_train)
                y_train_pred = nnModel.predict(x_train)
                y_train_accuracy = accuracy_score(y_train, y_train_pred)
                y_test_pred = nnModel.predict(x_test)
                y_test_accuracy = accuracy_score(y_test, y_test_pred)
                optTuning['training_size'].append(len(x_train))
                   
                optTuning['training_accuracy'].append(y_train_accuracy)
                optTuning['test_accuracy'].append(y_test_accuracy)
                cv_results = cross_validate(nnModel,x_train, y_train, cv= 5)
                optTuning['cv_training_accuracy'].append(np.mean(cv_results['train_score']))
                optTuning['cv_test_accuracy'].append(np.mean(cv_results['test_score']))
                optTuning['algorithm'].append(nnOptAlgos[y])
                optTuning['iterations'].append(i)
                optTuning['attempts'].append(j)
                
   
    
    optTuning = pd.DataFrame(optTuning)
    

   
    nnGraphs(optTuning, nnOptAlgos)
def nnGraphs(df, algoNames):
  
    for i in range(3):
        
        if 'iterations' in df.columns:
            results2 = pd.pivot_table(df[(df.algorithm == algoNames[i])], values='test_accuracy', index='iterations', columns='attempts')
            results3 = pd.pivot_table(df[(df.algorithm == algoNames[i])], values='cv_test_accuracy', index='iterations', columns='attempts')    
            plt.title(algoNames[i] + ', ' + 'Test Accuracy')
            ax1 = sns.heatmap(results2, cmap = 'Greens')
            plt.savefig(algoNames[i] + ', ' + 'Training Size (best test accuracy).png')
            plt.clf()
            plt.figure()
            plt.title(algoNames[i] + ', ' + 'CV Test Accuracy')
            ax2 = sns.heatmap(results3, cmap = 'YlGnBu')
            plt.savefig(algoNames[i] + ', ' + 'Training Size (best CV test accuracy).png')
            plt.clf()
            plt.figure()
            
        else:   
            results = df[(df.algorithm == algoNames[i])]
            
            plt. title(algoNames[i])
            plt.plot(results['training_size'], results['training_accuracy'], label='Training Accuracy')
            plt.plot(results['training_size'], results['test_accuracy'], label = 'Test Accuracy')
            plt.plot(results['training_size'], results['cv_training_accuracy'], label = 'CV Training Accuracy')
            plt.plot(results['training_size'], results['cv_test_accuracy'], label = 'CV Test Accuracy')
            plt.xlabel('Training Size')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(algoNames[i]+ ' Training size vs Accuracy' +'.png')
            
            plt.clf()
            plt.figure()
       
           
        
    
  
    
def optimizationFunction(n,iMin, iMax, iStep, jMin, jMax, jStep):
    
    
    np.random.seed(100)
    optScenarios = ['Knap Sack', 'Four Peaks', 'K - Colors']
    for x in range(3):
        if x == 0:
            weights = np.random.randint(1,10,size=n)
            #values = np.random.randint(1,50,size=n)
            values = [i for i in range(1,n+1)]
            max_weight_pct = 0.5
            fitnessFunction = mlrose.Knapsack(weights, values, max_weight_pct)
            optModel = mlrose.DiscreteOpt(len(values), fitness_fn = fitnessFunction, maximize=True)
        elif x == 1:
            inp = [0] * int(n/2) + [1]*int(n - int(n/2))
            np.random.shuffle(inp)
            fitnessFunction = mlrose.FourPeaks(t_pct = 0.15)
            optModel = mlrose.DiscreteOpt(len(inp), fitness_fn = fitnessFunction, maximize =True)
        elif x == 2:
            edges = [(np.random.randint(0,n), np.random.randint(0,n)) for ab in range(n)]
            fitnessFunction = mlrose.MaxKColor(edges)
            optModel = mlrose.DiscreteOpt(len(edges), fitness_fn = fitnessFunction, maximize =True)
                
        decay = mlrose.ExpDecay()
    

        optResults = {'iterations':[],'attempts':[],'fitness':[],'time':[], 'optimization':[]}
        for i in range(iMin,iMax,iStep):

            for j in range(jMin,jMax,jStep):
                start_time = timer()
                best_state, best_fitness = mlrose.random_hill_climb(optModel, max_attempts = j, max_iters = i, random_state=100)
                opt_time = timer() - start_time
           
                optResults['iterations'].append(i)
                optResults['attempts'].append(j)
                optResults['fitness'].append(best_fitness)
                optResults['time'].append(opt_time)
                optResults['optimization'].append('Random Hill')
                start_time = timer()
                best_state, best_fitness = mlrose.simulated_annealing(optModel, schedule=decay, max_attempts = j,max_iters = i,random_state=1000)
                opt_time = timer() - start_time
                optResults['iterations'].append(i)
                optResults['attempts'].append(j)
                optResults['fitness'].append(best_fitness)
                optResults['time'].append(opt_time)
                optResults['optimization'].append('Simulated Annealing')
           
                start_time = timer()
                best_state, best_fitness = mlrose.genetic_alg(optModel, pop_size=200, mutation_prob = 0.25, max_attempts = j, max_iters = i, random_state=5000)
                opt_time = timer() - start_time
               
                optResults['iterations'].append(i)
                optResults['attempts'].append(j)
                optResults['fitness'].append(best_fitness)
                optResults['time'].append(opt_time)
                optResults['optimization'].append('Genetic Algorithm')
                start_time = timer()
                best_state, best_fitness = mlrose.mimic(optModel, pop_size = 200, keep_pct = 0.3, max_attempts = j, max_iters = i, random_state=150)
                opt_time = timer() - start_time
                optResults['iterations'].append(i)
                optResults['attempts'].append(j)
                optResults['fitness'].append(best_fitness)
                optResults['time'].append(opt_time)
                optResults['optimization'].append('MIMIC')
       
        optResults = pd.DataFrame(optResults)
 
        plotGraphs(optResults,optScenarios[x])
 

def plotGraphs(df,n1):
    optNames = ['Random Hill', 'Simulated Annealing', 'Genetic Algorithm', 'MIMIC']
    colorMap = ['PuBuGn', 'YlOrBr', 'YlOrRd','YlGnBu']
   
    for i in range(4):
        results = pd.pivot_table(df[df.optimization==optNames[i]], values='fitness', index='iterations', columns='attempts')
   
        plt.title(n1 + ', ' + optNames[i] + ' -> Fitness')
        ax1 = sns.heatmap(results, cmap=colorMap[i])
        plt.savefig(n1 + ', ' + optNames[i] + ' -> Fitness.png')
        plt.clf()
        plt.figure()
      
     
     
        plt.title(n1 + ', ' + optNames[i] + ' -> Time')
        resultsTime = pd.pivot_table(df[df.optimization==optNames[i]], values='time', index='iterations', columns='attempts')
        ax2 = sns.heatmap(resultsTime, cmap=colorMap[i])
        plt.savefig(n1 + ', ' + optNames[i] + ' -> Time.png')
        plt.clf()
        plt.figure()
        
        
    for i in range(4):
        resultsIterFitness = df[(df.attempts == df.loc[df.attempts.idxmin(),'attempts']) & (df.optimization == optNames[i])]
       
        plt.plot(resultsIterFitness['iterations'], resultsIterFitness['fitness'], label = optNames[i])
    plt.title(n1 + ' -> Fitness')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig(n1 + ' -> Fitness.png')
    plt.clf()
    plt.figure()
   
    for i in range(4):
         resultsIterFitness = df[(df.attempts == df.loc[df.attempts.idxmin(),'attempts']) & (df.optimization == optNames[i])]
         plt.plot(resultsIterFitness['iterations'], resultsIterFitness['time'], label =optNames[i])
    plt.title(n1 + ' -> Time')
    plt.xlabel('Iterations')
    plt.ylabel('Time')
    plt.legend()
    plt.savefig(n1 + ' -> Time.png')
    plt.clf()
    plt.figure()
 
        
if __name__ == "__main__":
    optimizationFunction(30,1,102,20,5,30,5)
   
    nnModelFunction(1,52,10,1,25,5,0.1,0.9,0.1)
    