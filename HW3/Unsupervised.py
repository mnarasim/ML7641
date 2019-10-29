#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:31:16 2019

@author: Mani
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score,homogeneity_score,completeness_score,fowlkes_mallows_score 
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn import neural_network
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import accuracy_score

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


def unSupervised(x_data,y_data, x,n):
    fNames = ['Heart', 'Credit Card']

            
    clusterValues = []
    silScores = []
    noComponents = []
    bic = []
    aic = []
  
    arScore = []
    amiScore = []
    homogeneityScore=[]
    completenessScore=[]
    fmScore = []
    
    for a in range(2,7):
        ## K-means
        kmeans = KMeans(n_clusters = a)
        kmeans.fit(x_data)
        kmeans.predict(x_data)
        labels = kmeans.labels_
        silScore = silhouette_score(x_data,labels)
        
       
        clusterVisuals(x_data, labels, a, fNames[x] +': ' + n + ': K-Means')
  
        clusterValues.append(a)
        silScores.append(silScore)
        arScore.append(adjusted_rand_score(y_data,labels))
        amiScore.append(adjusted_mutual_info_score(y_data, labels))
        homogeneityScore.append(homogeneity_score(y_data, labels))
        completenessScore.append(completeness_score(y_data, labels))
        fmScore.append(fowlkes_mallows_score(y_data, labels))
        
        ###Expected maximization
        em = GaussianMixture(n_components = a)
        em.fit(x_data)
        labels = em.predict(x_data)
        
        noComponents.append(a)
        bic.append(em.bic(x_data))
        aic.append(em.aic(x_data))
        clusterVisuals(x_data, labels, a, fNames[x] + ': ' + n + ': EM')
            
    plt.plot(clusterValues, silScores, label ='Silhouette')
    plt.plot(clusterValues,arScore, label='Adjusted Rand Index')
    plt.plot(clusterValues,amiScore, label='Ajusted Mutual Index')
    plt.plot(clusterValues,homogeneityScore, label='Homogeneity')
    plt.plot(clusterValues,completenessScore, label='Completeness')
    plt.plot(clusterValues,fmScore, label='Fowlkes-Mallows')
    plt.xlabel('No. of Clusters')
    plt.ylabel('Scores')
    plt.legend()
    plt.title(n + ': K-Means: ' + fNames[x])
    plt.savefig(n + ' K-Means ' + fNames[x] +'.png')
    plt.figure()
    
    plt.title(n + ': ' + 'Expected Maximzation: ' + fNames[x])
    plt.plot(noComponents, bic, label="BIC")
    plt.plot(noComponents, aic, label = "AIC")
    plt.xlabel("No. of Components")
    plt.ylabel("BIC & AIC")
    plt.legend()
    plt.savefig(n + ' ' + 'Expected Maximzation: ' + fNames[x] +'.png')
    plt.figure()
        

      
def dimensionReduction(x_data,y_data, x):
    fNames = ['Heart', 'Credit Card']

    dNames = ['PCA', 'ICA', 'Gaussian Random', 'SVD']
    silScores = []
   
    bic = []
    aic = []
  
    arScore = []
    amiScore = []
    homogeneityScore=[]
    completenessScore=[]
    fmScore = []
        
        
    
    #PCA
 
    z = dRuns(x_data, 'PCA')   

    pca = PCA(n_components=z)
    newData = pca.fit_transform(x_data)

    varRatio = pca.explained_variance_ratio_
    plt.bar(range(1, len(varRatio)+1),varRatio, label = 'Individual Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance')
    plt.title(fNames[x] + ': PCA')
    plt.legend()
    plt.savefig(fNames[x] + ': PCA.png')
    plt.figure()
        
    #unSupervised(newData,y_data, x,'PCA')
    a,b,c,d,e,f,g,h = dimReducedClusters(newData, y_data,x, 'PCA')
    silScores.append(a)
    arScore.append(b)
    amiScore.append(c)
    homogeneityScore.append(d)
    completenessScore.append(e)
    fmScore.append(f)
    bic.append(g)
    aic.append(h)
    
    if x == 0:
        nNetwork(newData, y_data, 'PCA')
    
    #ICA
    z = dRuns(x_data, 'ICA')
    ica = FastICA(n_components=z)
    newData = ica.fit_transform(x_data)
    newDatadf = pd.DataFrame(newData)
    icaKurt = newDatadf.kurt(axis=0)    
      
      
       
    plt.plot(range(1, newDatadf.shape[1]+1),icaKurt, label='Kurtosis')
    plt.xlabel('Component')
    plt.ylabel('Kurtosis')
  
    plt.title(fNames[x] + ': ICA Kurtosis')
    plt.legend()
    plt.savefig(fNames[x] + ': ICA Kurtosis.png')
    plt.figure()
    #unSupervised(newData,y_data, x,'ICA')
    
    a,b,c,d,e,f,g,h = dimReducedClusters(newData, y_data,x, 'ICA')
    silScores.append(a)
    arScore.append(b)
    amiScore.append(c)
    homogeneityScore.append(d)
    completenessScore.append(e)
    fmScore.append(f)
    bic.append(g)
    aic.append(h)
    
    if x == 0:
        nNetwork(newData, y_data, 'ICA')
    #Randomized Projections
    z = dRuns(x_data, 'Random')  
    randProjection = GaussianRandomProjection(n_components=z)
    newData = randProjection.fit_transform(x_data)
    newDatadf = pd.DataFrame(newData)
    grpKurt = newDatadf.kurt(axis=0)
    grpSkew = newDatadf.skew(axis=0)
    plt.plot(range(1, newDatadf.shape[1]+1),grpKurt, label='Kurtosis')
    plt.plot(range(1, newDatadf.shape[1]+1),grpSkew, label='Skewness')
    plt.xlabel('Component')
    plt.ylabel('Kurtosis and Skewness')
    plt.legend()
    plt.title(fNames[x] + ': Gaussian Random - Kurtosis & Skewness')
    plt.savefig(fNames[x] + ': Gaussian Random - Kurtosis & Skewness.png')
    plt.figure()
        
    #unSupervised(newData,y_data, x,'Random Gaussian')
    
    a,b,c,d,e,f,g,h = dimReducedClusters(newData, y_data,x, 'Random Gaussian')
    silScores.append(a)
    arScore.append(b)
    amiScore.append(c)
    homogeneityScore.append(d)
    completenessScore.append(e)
    fmScore.append(f)
    bic.append(g)
    aic.append(h)
    if x == 0:
        nNetwork(newData, y_data, 'Random Gaussian')
    #Other
    z = dRuns(x_data, 'SVD')   
    svd = TruncatedSVD(n_components = z)
    newData = svd.fit_transform(x_data)
    svdVarRatio = svd.explained_variance_ratio_
   
    plt.bar(range(1, len(svdVarRatio)+1),svdVarRatio, label = 'Individual Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance')
    plt.title(fNames[x] + ': SVD')
    plt.savefig(fNames[x] + ': SVD.png')
    plt.legend()
    plt.figure()
    #unSupervised(newData,y_data, x,'SVD')
    
    a,b,c,d,e,f,g,h = dimReducedClusters(newData, y_data,x, 'SVD')
    silScores.append(a)
    arScore.append(b)
    amiScore.append(c)
    homogeneityScore.append(d)
    completenessScore.append(e)
    fmScore.append(f)
    bic.append(g)
    aic.append(h)
    
    if x == 0:
        nNetwork(newData, y_data, 'SVD')
        
    #########################
    plt.plot(dNames, silScores, label ='Silhouette')
    plt.plot(dNames,arScore, label='Adjusted Rand Index')
    plt.plot(dNames,amiScore, label='Ajusted Mutual Index')
    plt.plot(dNames,homogeneityScore, label='Homogeneity')
    plt.plot(dNames,completenessScore, label='Completeness')
    plt.plot(dNames,fmScore, label='Fowlkes-Mallows')
    plt.xlabel('Dimensionality Reduction Technique')
    plt.ylabel('Scores')
    plt.legend()
    plt.title('K-Means: ' + ' ' + fNames[x])
    plt.savefig('K-Means' + ' ' + fNames[x] +'.png')
    plt.figure()
    
    plt.title('Expected Maximzation: ' + fNames[x])
    plt.plot(dNames, bic, label="BIC")
    plt.plot(dNames, aic, label = "AIC")
    plt.xlabel("Dimensionality Reduction Technique")
    plt.ylabel("BIC & AIC")
    plt.legend()
 
    plt.savefig('Expected Maximzation: ' + fNames[x] +'.png')
    plt.figure()
    
    
def nNetwork(x_data, y_data, n):
    np.random.seed(1000)
    classificationModel = neural_network.MLPClassifier()
    trainAccuracyResults = []
    testAccuracyResults = []
    trainingSize = []
    cvTrainAccuracy = []
    cvTestAccuracy = []
    for b in np.arange(0.1, 0.9,0.1):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = b , random_state =100, shuffle = True)
        classificationModel.fit(x_train, y_train)
        modelTrain = classificationModel.predict(x_train)
        modelPredict = classificationModel.predict(x_test)
        modelTrainAccuracy = accuracy_score(y_train, modelTrain)
        modelTestAccuracy = accuracy_score(y_test, modelPredict)
                
        trainAccuracyResults.append(modelTrainAccuracy)
        testAccuracyResults.append(modelTestAccuracy)
        trainingSize.append(len(x_train))
        cv_results = cross_validate(classificationModel,x_train, y_train, cv= 5)
        cvTrainAccuracy.append(np.mean(cv_results['train_score']))
        cvTestAccuracy.append(np.mean(cv_results['test_score']))
    plt.plot(trainingSize, trainAccuracyResults, label = 'Training Accuracy')
    plt.plot(trainingSize, testAccuracyResults, label = 'Test Accuracy')
    plt.plot(trainingSize, cvTrainAccuracy, label = 'CV Train Accuracy')
    plt.plot(trainingSize, cvTestAccuracy, label = 'CV Test Accuracy')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(n + ': Training Size vs Accuracy')
    plt.savefig(n + ': Training Size vs Accuracy.png')
   
    plt.figure()

def dimReducedClusters(x_data, y_data, x,n):
    
     kmeans = KMeans(n_clusters = 2)
     kmeans.fit(x_data)
     kmeans.predict(x_data)
     kLabels = kmeans.labels_
     
     if x == 0:
        nData = pd.DataFrame(x_data)
        nData['cluster'] = kLabels
        nNetwork(nData,y_data, n + ': After K-Means')
        
     em = GaussianMixture(n_components = 2)
     em.fit(x_data)
     eLabels = em.predict(x_data)
     if x == 0:
         nData = pd.DataFrame(x_data)
         nData['cluster'] = eLabels
         nNetwork(nData,y_data, n +': After EM')
     
     a = silhouette_score(x_data,kLabels)
     b = adjusted_rand_score(y_data,kLabels)
     c = adjusted_mutual_info_score(y_data, kLabels)
     d = homogeneity_score(y_data, kLabels)
     e = completeness_score(y_data, kLabels)
     f = fowlkes_mallows_score(y_data, kLabels)
     g = em.bic(x_data)
     h = em.aic(x_data)
        
     return a,b,c,d,e,f,g,h
 
def dRuns(x_data, n):
    k1 = []

 
    if n == 'PCA' or n == 'SVD':
        for z in range(2,np.shape(x_data)[1]+1):
            if n == 'PCA':
                pca = PCA(n_components=z)
                newData = pca.fit_transform(x_data)
                varRatio = pca.explained_variance_ratio_
            elif n == 'SVD':
                svd = TruncatedSVD(n_components = z)
                newData = svd.fit_transform(x_data)
                varRatio = svd.explained_variance_ratio_
            if np.sum(varRatio) >= 0.80:
                return z
                
    if n == 'ICA' or n == 'Random':
        for z in range(2,np.shape(x_data)[1]+1):
            if n == 'ICA':
                ica = FastICA(n_components=z)
                newData = ica.fit_transform(x_data)
                newData = pd.DataFrame(newData)
                k1.append(np.mean(newData.kurt()))
            else:
                randProjection = GaussianRandomProjection(n_components=z)
                newData = randProjection.fit_transform(x_data)
                newData = pd.DataFrame(newData)
                k1.append(np.mean(newData.kurt()))
        
        return np.argmax(k1)+2
            
            
def clusterVisuals(x_data, labels,a, n):        
    visual = pd.DataFrame(x_data)

    visual['cluster'] = labels
    visual_TSNE = TSNE(n_components=2).fit_transform(x_data)
    vis_x = visual_TSNE[:,0]
    vis_y = visual_TSNE[:,1]
    plt.scatter(vis_x, vis_y, c=labels)
    plt.title(n + ' ' + ', No. of Clusters :' + str(a))
    plt.savefig(n + ' ' + ', No. of Clusters ' + str(a) +'.png')
       
    plt.figure()    
      
    
if __name__ == "__main__":
    for x in range(2):
        data = readFile(x)
        x_data = data.iloc[:,0:-1].values
        y_data = data.iloc[:,-1].values
        x_data = scaledData(x_data)
        unSupervised(x_data,y_data, x, 'Base')
    for x in range(2):
        data = readFile(x)
        x_data = data.iloc[:,0:-1].values
        y_data = data.iloc[:,-1].values
        x_data = scaledData(x_data)
        dimensionReduction(x_data, y_data, x)