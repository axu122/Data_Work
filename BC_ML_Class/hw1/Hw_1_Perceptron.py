#!/usr/bin/env python
# coding: utf-8

# # 4a 

# In[ ]:


#Import neccesary packages
from sklearn.linear_model import Perceptron
import numpy as np
import pandas as pd


# In[ ]:


#Loading the data
dataset_1 = np.loadtxt('sampleData1.txt',delimiter=',')
(numSamples_1, numFeatures_1) = dataset_1.shape
data_1 = dataset_1[:,range(2)].reshape((numSamples_1, 2))
labels_1 = dataset_1[:, 2].reshape((numSamples_1,))

dataset_2 = np.loadtxt('sampleData2.txt',delimiter=',')
(numSamples_2, numFeatures_2) = dataset_2.shape
data_2 = dataset_2[:,range(2)].reshape((numSamples_2, 2))
labels_2 = dataset_2[:, 2].reshape((numSamples_2,))

dataset_3 = np.loadtxt('sampleData3.txt',delimiter=',')
(numSamples_3, numFeatures_3) = dataset_3.shape
data_3 = dataset_3[:,range(2)].reshape((numSamples_3, 2))
labels_3 = dataset_3[:, 2].reshape((numSamples_3,))


# In[ ]:


#load perceptron
#generate stop counter
#generate counter to track weight changes 
def partial_1():
    perceptron_1 = Perceptron()
    stop_1 = 0
    counter_1 = 0
    while stop_1 == 0:
        for x in range(1000):
            perceptron_1.partial_fit([data_1[x]],[labels_1[x]], classes= np.unique(labels_1))
            if perceptron_1.score(data_1,labels_1) == 1:
                counter_1 +=1
                stop_1 +=1
                break
            else:
                counter_1 +=1
    weights = perceptron_1.coef_
    w1 = weights[0][0]
    w2 = weights[0][1]
    w0 = perceptron_1.intercept_[0]
    print("Weights was adjusted {} times".format(counter_1))
    print("Intercept (w0) is {}".format(w0))
    print("Final weight vector is : {} Hence:".format(weights))
    print("w1 is : {}  , w2 weight is: {}".format(w1, w2))
    #deriving the line based on HW Problem 1.2
    a = -1* (w1/w2)
    b = -1* (w0/w2)
    print("The equation of the decision boundary  line is y = ({})x + ({})".format(a,b))


# In[ ]:


def partial_2():
    perceptron_2 = Perceptron()
    stop_2 = 0
    counter_2 = 0
    while stop_2 == 0:
        for x in range(1000):
            perceptron_2.partial_fit([data_2[x]],[labels_2[x]], classes= np.unique(labels_2))
            if perceptron_2.score(data_2,labels_2) == 1:
                counter_2 +=1
                stop_2 +=1
                break
            else:
                counter_2 +=1
    weights = perceptron_2.coef_
    w1 = weights[0][0]
    w2 = weights[0][1]
    w0 = perceptron_2.intercept_[0]
    print("Weights was adjusted {} times".format(counter_2))
    print("Intercept (w0) is {}".format(w0))
    print("Final weight vector is : {} Hence:".format(weights))
    print("w1 is : {}  , w2 weight is: {}".format(w1, w2))
    #deriving the line based on HW Problem 1.2
    a = -1* (w1/w2)
    b = -1* (w0/w2)
    print("The equation of the decision boundary line is y = ({})x + ({})".format(a,b))


# In[ ]:


def partial_3():
    perceptron_3 = Perceptron()
    stop_3 = 0
    counter_3 = 0
    while stop_3 == 0:
        for x in range(1000):
            perceptron_3.partial_fit([data_3[x]],[labels_3[x]], classes= np.unique(labels_3))
            if perceptron_3.score(data_3,labels_3) == 1:
                stop_3 +=1
            if counter_3 >= 100000:
                return print('No Convergence')
            else:
                counter_3 +=1
    weights = perceptron_3.coef_
    w1 = weights[0][0]
    w2 = weights[0][1]
    w0 = perceptron_3.intercept_[0]
    print("Weights was adjusted {} times".format(counter_3))
    print("Intercept (w0) is {}".format(w0))
    print("Final weight vector is : {} Hence:".format(weights))
    print("w1 is : {}  , w2 weight is: {}".format(w1, w2))
    #deriving the line based on HW Problem 1.2
    a = -1* (w1/w2)
    b = -1* (w0/w2)
    print("The equation of the decision boundary line is y = ({})x + ({})".format(a,b))


# In[6]:


#partial_1()


# In[7]:


#partial_2()


# In[8]:


#partial_3()


# # 4b

# Main differences:
# 
# sampleData3.txt did not converge, which indicates that the points plotted on to a cartesians plane cannot be seperated by a linear line.
# 
# As for the main differences between sampleData2.txt and sampleData1.txt, due to the high volume of adjustments in sampleData2.txt, we can assume that the points of true classification and false classifications are tightly plotted. This would account for why the weights had to be adjusted so many times.
# 
# w2 has consistently been greater than w1 for the data we tested, which may mean that the second feature played a higher role in determinging the right classification. This seemed to be especially true for sampleData2.txt, with the intercept being 14, we can assume that the line is much higher, which would also explain the high volume of weight adjustments since adjustments are only increased or decreased by a minimal value.
# 
# The likely reason why sample_3 doesn't converge is because there doesn't exist a linear line such that positive and negative feedbacks can be distinctly seperated.

# In[9]:


#Sample 1:
partial_1()


# In[10]:


#Sample 2:
partial_2()


# In[11]:


#Sample 3:
partial_3()


# In[ ]:




