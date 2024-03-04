import argparse
import numpy as np
import pandas as pd
from graphviz import Digraph
from time import sleep

#correct
def calculate_score(y, criterion):
    """
    Given a numpy array of labels associated with a node, y, 
    calculate the score based on the crieterion specified.

    Parameters
    ----------
    y : numpy.1d array with shape (n, )
        Array of labels associated with a node
    criterion : String
        The function to measure the quality of a split.
        Supported criteria are "gini" for the Gini impurity
        and "entropy" for the information gain.
    Returns
    -------
    score : float
        The gini or entropy associated with a node
    """
    def gini_score(y):
        n = len(y)
        if n == 0:
            return 0
        p = np.bincount(y) / n
        return 1 - np.sum(p ** 2) 
    
    def entropy_score(y):
        n = len(y)
        if n == 0:
            return 0
        p = np.bincount(y) / n
        if 0 in p:
            return 0
        return -np.sum(p * np.log2(p))

    if criterion == "gini":
        return gini_score(y)
    if criterion == "entropy":
        return entropy_score(y)
    return None    

def find_best_splitval(xcol, y, criterion, minLeafSample):
    """
    Given a feature column (i.e., measurements for feature d),
    and the corresponding labels, calculate the best split
    value, v, such that the data will be split into two subgroups:
    xcol <= v and xcol > v. If there is a tie (i.e., multiple values
    that yield the same split score), you can return any of the
    possible values.

    Parameters
    ----------
    xcol : numpy.1d array with shape (n, )
    y : numpy.1d array with shape (n, )
        Array of labels associated with a node
    criterion : string
        The function to measure the quality of a split.
        Supported criteria are "gini" for the Gini impurity
        and "entropy" for the information gain.
    minLeafSample : int
            The min
    Returns
    -------
    v:  float / int (depending on the column)
        The best split value to use to split into 2 subgroups.
    score : float
        The gini or entropy associated with the best split
    """    
    best_splitval=None
    best_score=1
    
    len_y = len(y)
    if (len_y <=minLeafSample*2):
        return None, 2
    
    temp = np.argsort(xcol)
    y=y[temp]
    xcol=xcol[temp]
    
    for i in range(minLeafSample, len_y-minLeafSample):
        if xcol[i]!=xcol[i-1]:
            left= y[xcol<=xcol[i]]
            right= y[xcol>xcol[i]]
            len_left=len(left)
            len_right=len(right)
            current_score=(len_left/len_y)*calculate_score(left, criterion) + (len_right/len_y)*calculate_score(right, criterion)
            if current_score<best_score:
                best_score=current_score
                best_splitval=xcol[i]
    return best_splitval, best_score
            
            
def calculate_label(y):
    """
    Given a numpy array of labels associated with a node, y, 
    calculate the label that should be assigned to the node.

    Parameters
    ----------
    y : numpy.1d array with shape (n, )
        Array of labels associated with a node

    Returns
    -------
    label : int
        The label to assign to the node
    """
    return np.argmax(np.bincount(y))
    

class Node(object):
    left = None
    right = None
    splitFeat = None
    splitVal = None
    leaf = False
    label = None
    
    def __init__(self, label=None):
        """
        Node constructor

        Parameters
        ----------
        splitFeat : int
            The feature to split on
        splitVal : float
            The value to split on
        """
        self.label = label
        


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    startNode = None   # root node of the decision tree

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters 
        ----------
        xFeat : numpy nd-array with shape (n, d)
            Training data 
        y : numpy 1d array with shape (n, )
            Array of labels associated with training data.

        Returns
        -------
        self : DecisionTree
            The decision tree model instance
        """
        startlabel = calculate_label(y)
        self.startNode = Node(startlabel)
        
        def build_tree(node, xFeat, y, depth):
            if depth>=self.maxDepth or len(y)<=self.minLeafSample:
                node.leaf=True
                node.label=calculate_label(y)  
                return
            node.label=calculate_label(y) 
            
            
            best_score = 1
            best_split = None
            best_splitval = None
            for i in range(xFeat.shape[1]):
                splitval, score = find_best_splitval(xFeat[:,i].flatten(), y, self.criterion, self.minLeafSample)
                if score < best_score:
                    # print("score:", score, "best_score:", best_score, "splitval:", i)
                    best_score = score
                    best_split = i
                    best_splitval = splitval
            if best_split is None:
                node.leaf = True
                return
            
            node.left=Node()
            node.right=Node()
            node.splitFeat = best_split
            node.splitVal = best_splitval
            
            Left = np.where(xFeat[:,best_split] <= best_splitval)
            Right = np.where(xFeat[:,best_split] > best_splitval)
            
            if len(Left[0])<=self.minLeafSample or len(Right[0])<=self.minLeafSample:
                node.leaf=True
                node.label=calculate_label(y)
                return
            
            left_node=Node()
            right_node=Node()
            node.left=left_node
            node.right=right_node
            build_tree(left_node, xFeat[Left], y[Left], depth+1)
            build_tree(right_node, xFeat[Right], y[Right], depth+1)
        
        build_tree(self.startNode, xFeat, y, 0)
        
        return self

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (m, d)
            The data to predict.  

        Returns
        -------
        yHat : numpy 1d array with shape (m, )
            Predicted class label per sample
        """
        yHat = np.zeros(xFeat.shape[0])
        for i in range(xFeat.shape[0]):
            node=self.startNode
            feat=xFeat[i]
            while (node.leaf != True):
                if(feat[node.splitFeat]<=node.splitVal):
                    node=node.left
                elif(feat[node.splitFeat]>node.splitVal):
                    node=node.right
            yHat[i]=node.label   
        
        return yHat
    
    def visualize_tree(self, node=None, depth=0, parent_name=None):
        if node is None:
            node = self.startNode
            self.dot = Digraph(comment='The Decision Tree')
        
        node_name = f'Node{depth}_{id(node)}'
        if node.leaf:
            self.dot.node(node_name, label=f'Label: {node.label}', shape='box')
        else:
            self.dot.node(node_name, label=f'label: {node.label}\nfeature {node.splitFeat}<={node.splitVal}')
        
        if parent_name:
            self.dot.edge(parent_name, node_name)
        
        if node.left:
            self.visualize_tree(node.left, depth + 1, node_name)
        
        if node.right:
            self.visualize_tree(node.right, depth + 1, node_name)
        
        return self.dot

def _accuracy(yTrue, yHat):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yTrue : 1d-array with shape (n, )
        True labels associated with the n samples
    yHat : 1d-array with shape (n,)
        Predicted class label for n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    acc = np.sum(yHat == yTrue) / len(yTrue)
    return acc

def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : numpy.nd-array with shape n x d
        Training data 
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data.
    xTest : numpy.nd-array with shape m x d
        Test data 
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = _accuracy(yTrain, yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = _accuracy(yTest, yHatTest)
    return trainAcc, testAcc