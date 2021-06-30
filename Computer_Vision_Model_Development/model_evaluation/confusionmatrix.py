#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:49:29 2017

@author: anmo
"""

import numpy as np
import os.path

class confusionmatrix:
    
    # Rows correspond to true labels
    # Columns correspond to predicted labels
    
    def __init__(self, numClasses):
        self.numClasses = numClasses
        self.confMat = np.zeros(shape=(numClasses,numClasses))
        
    def __str__(self):
        # Create and return string for print()
        return np.array2string(self.confMat/self.trueCounts()[:,None], max_line_width=1000,precision=3, suppress_small=True)
    
    def Append(self, trueLabels, predictedLabels):

        # Convert to numpy arrays
        trueLabels = np.asarray(trueLabels)
        predictedLabels = np.asarray(predictedLabels)
        
        # Size of true labels and predicted labels must be the same
        assert(trueLabels.shape == predictedLabels.shape)
        
        # Setup temp confusion matrix
        confMat_tmp = np.zeros(shape=(self.numClasses,self.numClasses))
        for t in range(self.numClasses):
            # Get index of all true labels belonging to class 't'
            tIdx = np.where(np.in1d(trueLabels, t))
            for p in range(self.numClasses):
                # Get index of all predictions belonging to class 'p'
                pIdx = np.where(np.in1d(predictedLabels, p))
                # Find the indicies where both true label belongs to 't' and prediction belongs to 'p'
                intersection = np.intersect1d(tIdx,pIdx)
                # Store 
                confMat_tmp[t][p] = len(intersection)

        # Add temp confusion matrix to instance confusion matrix
        self.confMat = self.confMat + confMat_tmp
        return self
    
    def Reset(self):
        # Set set all elemants in instance confusion matrix to 0
        self.confMat = np.zeros(shape=(self.numClasses, self.numClasses))
        return self
    
    def Save(self, file, fileFormat=None):
        # Save the confusion matrix into a binary file in Numpy *.npy format
        
        # Determine fileformat is not specified
        if (fileFormat == None):
            filename, fileExtension = os.path.splitext(file)
            fileExtension = fileExtension[1:]
            if fileExtension in {'csv','txt','dat'}:
                fileFormat = 'csv'
            elif fileExtension in {'npy'}:
                fileFormat = 'npy'
            elif fileExtension in {'npz'}:
                fileFormat = 'npz'
            else:
                raise ValueError('Could not determine the appropriate file format from the specified file.')

        # Save the confusion matrix using the appropriate numpy saver
        if (fileFormat == 'csv'):
            np.savetxt(file, self.confMat,delimiter=',',fmt='%d')
        elif (fileFormat == 'npy'):
            np.save(file, self.confMat, allow_pickle=False)
        elif (fileFormat == 'npz'):
            np.savez(file, confMat = self.confMat)
        else:
            raise ValueError('Unknown file format.')
        
    def Load(self, file, fileFormat=None):
        # Load a previously saved confusion matrix
        
        # Determine fileformat is not specified
        if (fileFormat == None):
            filename, fileExtension = os.path.splitext(file)
            fileExtension = fileExtension[1:]
            if fileExtension in {'csv','txt','dat'}:
                fileFormat = 'csv'
            elif fileExtension in {'npy'}:
                fileFormat = 'npy'
            elif fileExtension in {'npz'}:
                fileFormat = 'npz'
            else:
                raise ValueError('Could not determine the appropriate file format from the specified file.')

        # Load the confusion matrix using the appropriate numpy saver
        if (fileFormat == 'csv'):
            self.confMat = np.loadtxt(file, delimiter=',')
        elif (fileFormat == 'npy'):
            self.confMat = np.load(file, mmap_mode=None, allow_pickle=False)
        elif (fileFormat == 'npz'):
            data = np.load(file, mmap_mode=None, allow_pickle=False)
            self.confMat = data['confMat']
        else:
            raise ValueError('Unknown file format.')
        
    ## METRICS ##
    def count(self):
        # Number of samples
        return np.sum(self.confMat, axis=(0,1))
    
    def trueCounts(self):
        # Number of true labels from each class
        return np.sum(self.confMat, axis=1)

    def predictedCounts(self):
        # Number of predicted labels from each class
        return np.sum(self.confMat, axis=0)
    
    def trueFrequency(self):
        # True relative frequency of each class
        TC = self.trueCounts()
        return TC / np.sum(TC)

    def predictedFrequency(self):
        # Predicted relative frequency of each class
        PC = self.predictedCounts()
        return PC / np.sum(PC)
        
    def truePositives(self):
        # Number class of interest classified as class of interest
        return np.diagonal(self.confMat)
    
    def falseNegatives(self):
        # Number of class of interest classified as other classes
        return self.trueCounts() - self.truePositives()

    def falsePositives(self):
        # Number of other classes predicted as class of interest
        return self.predictedCounts() - self.truePositives()
        
    def trueNegatives(self):
        # Number of other classes classified as another class than the class of interest
        return np.subtract(self.count(), self.truePositives() + self.falsePositives() + self.falseNegatives())
        
    def truePositiveRates(self):
        # True positive rate of each class
        P = self.trueCounts()
        TP = self.truePositives()
        TPR = np.divide(TP,P)
        return TPR
    
    def precision(self):
        # Precision of each class
        # Number of true positive predictions divided by the total number of positive predictions
        TP = self.truePositives()
        TPFP = self.predictedCounts()
        return np.divide(TP,TPFP)
    
    def recall(self):
        # Recall of each class.
        # Same as true positive rate. See truePositiveRates()
        return self.truePositiveRates()
        
    def accuracy(self):
        # Calculate accuracy
        # sum of true predictions divided by all predictions
        return np.sum(np.diagonal(self.confMat))/np.sum(self.confMat)
        
    def class_accuracy(self):
        # Calculate accuracy
        # sum of true predictions divided by all predictions
        return np.diagonal(self.confMat)/np.sum(self.confMat,axis=1)
    
    def intersectionOverUnion(self):
        # Calculate intersection over union for each class
        # True prediction of a given class divided by the total number of predictions and true labels of that class
        iou = np.zeros(shape=(self.numClasses))
        for t in range(self.numClasses):
            union = (np.sum(self.confMat[:,t]) + np.sum(self.confMat[t,:]) - self.confMat[t,t])
            intersection = self.confMat[t,t]
            if (union == 0):
                iou[t] = 0
            else:   
                iou[t] = intersection / union
        return iou
    
    def jaccardIndex(self):
        # Same as intersection over union. See confusionmatrix.intersectionOverUnion()
        # Has a close relationship to Dice's coefficient
        return self.intersectionOverUnion()
        
    def fScore(self, beta=1):
        P = self.precision()
        R = self.recall()
        return (1+beta*beta)*(P*R)/(beta*beta*P + R)
        
    def dicesCoefficient(self):
        # Same as F1-score and has a close relationship to Jaccard index
        J = self.jaccardIndex()
        return np.divide(2*J,1+J)