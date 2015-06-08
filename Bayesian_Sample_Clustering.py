'''This class implements single-linkage clustering based on the Chan-Darwiche 
distance between predictive distributions. Theoretical information on how to 
use this hierarchical clustering scheme can be found in the master's thesis 
Bayesian Hierarchic Sample Clustering by Gabriel Isheden.'''

import numpy as np
import copy as cp
from scipy.cluster.hierarchy import dendrogram
from sets import Set
import matplotlib.pyplot as plt


class Bayesian_Sample_Clustering:
    def __init__(self, dataPartition_npArray, FLATTENING_CONST, 
                priorEstimate_npArray, labels_list=None):
        '''Implements Bayesian_Sample_Clustering based on a dataPartition on a
        given dataset. '''
        
        
        self.FLATTENING_CONST = FLATTENING_CONST
        
        self.partition_npArray = dataPartition_npArray
        self.initialPartition_list = labels_list
        if labels_list == None:
            self.initialPartition_list = range(0,np.shape(self.partition_npArray)[1])
        self.partition_list = self.initialPartition_list
        self.priorEstimate_npArray = priorEstimate_npArray
        
        self.clusters_list = [[n_int] for n_int,item in enumerate(
                                                self.initialPartition_list)]
        self.currentClusters_list = [n_int for n_int,item in enumerate(
                                                self.initialPartition_list)]
        self.clustIter_int = 0
        self.linkage_npArray = np.zeros((len(self.partition_list)-1,4))
        self.minDistLast_flt = -1
        self.minDistNew_flt = 0
        self.minIndex_tuple = (0,0)
        self.cdDist_npArray = np.zeros(
                        (len(self.partition_list),len(self.partition_list)))
        
    def calculate_hierarchy(self, FLATTENING_CONST=None, 
                            priorEstimate_npArray=None):
        '''Calculates a clustering hierarchy based on current data. '''
        
        
        self.partition_list = cp.copy(self.initialPartition_list)
        if priorEstimate_npArray != None:
            self.priorEstimate_npArray = priorEstimate_npArray
        if FLATTENING_CONST != None:
            self.FLATTENING_CONST = FLATTENING_CONST
        
        totalIterations_int = len(self.partition_list)-1
        self.clustIter_int = 0
        while(self.clustIter_int < totalIterations_int):
            self.probabilities_npArray = self.partition_npArray + \
                self.FLATTENING_CONST*np.transpose(self.priorEstimate_npArray)
            
            self.chan_darwiche_minIndex()
            self.merge()
            self.clustIter_int = self.clustIter_int + 1
            
    def chan_darwiche_minIndex(self):
        '''Computes the Chan-Darwiche distance based on the computed 
        probabilites. '''
        
        
        if self.clustIter_int == 0:
            row_int = 0
            column_int = 1
            minDist_flt = np.inf
            minIndex_tuple = (0,0)
            while column_int < len(self.currentClusters_list):
                row_int = 0
                while row_int < column_int:
                    parwiseQuotients_npArray = np.divide(
                                    self.probabilities_npArray[:,row_int], 
                                    self.probabilities_npArray[:,column_int])
                    self.cdDist_npArray[(row_int, column_int)] = np.round(
                                np.log(max(parwiseQuotients_npArray)) - \
                                np.log(min(parwiseQuotients_npArray)),4)
                    if minDist_flt > self.cdDist_npArray[(row_int, column_int)]:
                        minDist_flt = self.cdDist_npArray[(row_int, column_int)]
                        minIndex_tuple = (row_int, column_int)
                    row_int = row_int + 1
                column_int = column_int + 1
        
        if self.clustIter_int > 0:
            row_int = 0
            column_int = 1
            self.cdDist_npArray = \
                    np.delete(self.cdDist_npArray, self.minIndex_tuple[1], 1)
            self.cdDist_npArray = \
                    np.delete(self.cdDist_npArray, self.minIndex_tuple[1], 0)
                    
            while column_int < len(self.currentClusters_list):
                row_int = 0
                while row_int < column_int:
                    if row_int == self.minIndex_tuple[0] \
                                    or column_int == self.minIndex_tuple[0]:
                        parwiseQuotients_npArray = np.divide(
                                        self.probabilities_npArray[:,row_int], 
                                        self.probabilities_npArray[:,column_int])
                        self.cdDist_npArray[(row_int, column_int)] = np.round(
                                    np.log(max(parwiseQuotients_npArray)) - \
                                    np.log(min(parwiseQuotients_npArray)),4)
                    row_int = row_int + 1
                column_int = column_int + 1
        
        row_int = 0
        column_int = 1
        minDist_flt = np.inf
        minIndex_tuple = (0,0)
        while column_int < len(self.currentClusters_list):
            row_int = 0
            while row_int < column_int:
                if minDist_flt > self.cdDist_npArray[(row_int, column_int)]:
                    minDist_flt = self.cdDist_npArray[(row_int, column_int)]
                    minIndex_tuple = (row_int, column_int)
                row_int = row_int + 1
            column_int = column_int + 1
        self.minIndex_tuple = minIndex_tuple
        self.minDistNew_flt = minDist_flt
        
    def merge(self):
        '''Merges the two classes with least Chan-Darwiche distance. '''
        
        
        mergeMatrix_npArray = np.identity(len(self.currentClusters_list))
        
        mergeMatrix_npArray[:,self.minIndex_tuple[0]] = \
            mergeMatrix_npArray[:,self.minIndex_tuple[0]] +\
            mergeMatrix_npArray[:,self.minIndex_tuple[1]]
        mergeMatrix_npArray = \
            np.delete(mergeMatrix_npArray, self.minIndex_tuple[1], 1)
        self.linkage_npArray[self.clustIter_int,0] = \
                        self.currentClusters_list[self.minIndex_tuple[0]]
        self.linkage_npArray[self.clustIter_int,1] = \
                        self.currentClusters_list[self.minIndex_tuple[1]]
        self.currentClusters_list[self.minIndex_tuple[0]] = \
                max(self.currentClusters_list) + 1
        self.currentClusters_list.pop(self.minIndex_tuple[1])
        self.partition_npArray = np.dot(self.partition_npArray, \
                                    mergeMatrix_npArray)
        if self.clustIter_int > 0:
            if self.minDistLast_flt == self.minDistNew_flt:
                self.linkage_npArray[self.clustIter_int,2] = \
                                self.linkage_npArray[self.clustIter_int-1,2]
            else:
                self.linkage_npArray[self.clustIter_int,2] = \
                            self.linkage_npArray[self.clustIter_int-1,2] + 1
        else:
            self.linkage_npArray[self.clustIter_int,2] = self.clustIter_int + 1
        self.minDistLast_flt = self.minDistNew_flt
        self.linkage_npArray[self.clustIter_int,3] = \
                        sum(self.partition_npArray[:,self.minIndex_tuple[0]])
                        
    def plot(self):
        '''Plots the BSC hierarchical clustering scheme. '''
        
        
        plt.title("Bayesian Sample Clustering")
        dendrogram(self.linkage_npArray,
            color_threshold=1,
            labels=[str(item) for item in self.initialPartition_list],
            show_leaf_counts=True)
        plt.show()
        
        
