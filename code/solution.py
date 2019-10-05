import numpy as np 
from helper import *
'''
Homework2: logistic regression classifier
'''


def logistic_regression(data, label, max_iter, learning_rate):
    '''
	The logistic regression classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
	'''
    

    #1 set weights to 0
    d = data.shape[1]
    w = np.zeros((d,1)) #inital weight vector

    N = len(data)
    gt = 0
    v = 0
    
    #2 iterate for max_iter
    for _ in range(max_iter): 
        #3 calculate the gradient
        for i in range(len(data)):
            x = data[i]
            x = np.reshape(x, (d,1))
            y = label[i]
            w_x = np.dot(w.T, x)
            gt += (-y*x) * sigmoid(-y*w_x)   
        gt /= N
        
        #4 set move direction
        v = -gt
        
        #5 update weights
        w = w + (learning_rate * v)
        
    return w

def sigmoid(S):
    return (1/(1 + np.exp(-S)))


def thirdorder(data):
    '''
	This function is used for a 3rd order polynomial transform of the data.
	Args:
	data: input data with shape (:, 3) the first dimension represents 
		  total samples (training: 1561; testing: 424) and the 
		  second dimesion represents total features.

	Return:
		result: A numpy array format new data with shape (:,10), which using 
		a 3rd order polynomial transformation to extend the feature numbers 
		from 3 to 10. 
		The first dimension represents total samples (training: 1561; testing: 424) 
		and the second dimesion represents total features.
	'''

    data_thirdOrder = []
    for i in range(len(data)):
        x1 = data[i][0]
        x2 = data[i][1]
        row = [1,x1,x2, x1**(2), x1*x2, x2**(2), x1**(3), x1**(2) * x2, x1* x2**(2), x2**(3)]
        data_thirdOrder.append(row)
    
    return np.array(data_thirdOrder)[:,:]


def accuracy(x, y, w):
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    d = x.shape[1]
    pred_labels = []

    for i in range(len(x)):
        x_i = x[i]
        x_i = np.reshape(x_i, (d,1))
        prediction = sigmoid(np.dot(w.T, x_i))
        if prediction > .5:
            pred_labels.append(1)
        else:
            pred_labels.append(-1)
      
    pred_labels = np.array(pred_labels)[:]
    difference = pred_labels.T - y
    accuracy = 1.0 - (np.count_nonzero(difference) / len(y))
    return (accuracy *100) 
    



