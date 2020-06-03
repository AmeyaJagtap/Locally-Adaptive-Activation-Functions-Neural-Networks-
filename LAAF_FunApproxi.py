#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dr. Ameya D. Jagtap, Brown University, USA

Reference Article : A D Jagtap, K Kawaguchi, G E Karniadakis, Locally adaptive activation functions with slope recovery 
term for deep and physics-informed neural networks, arXiv preprint arXiv:1909.12228, 2019. 
(Accepted in Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences)
"""


""" Description:
    
The LAAF code for 1D function approximation using slope recovery term
"""
#%%
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
#from scipy.interpolate import griddata
from plotting import newfig, savefig

np.random.seed(1234)
tf.random.set_random_seed(1234)
#%%
def fun_x(x):
    
    f = np.zeros(len(x))
    f = np.reshape(f, (-1, 1))
    #f = []
    for i in range(len(x)):
        
        if x[i]<=0:
                  
            f[i] = 0.2*np.sin(6*x[i]) 
            
        else:
            
            f[i] = 0.1*x[i]*np.cos(18*x[i]) + 1
            
    return f


def hyper_parameters_A(size):

    a = tf.Variable(tf.constant(0.1, shape=size))

    return a

def hyper_parameters(size):

    return tf.Variable(tf.random_normal(shape=size, mean = 0., stddev = 0.1))

def DNN(X, W, b,a):
    A = X
    L = len(W)
    for i in range(L - 1):
        A = tf.tanh(10*a[i]*tf.add(tf.matmul(A, W[i]), b[i])) #
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y

if __name__ == "__main__":
   
    
    N = 300
    x = np.linspace(-3,3, N+1)
    x = np.reshape(x, (-1, 1))
    y = fun_x(x)

    layers = [1] + 4*[50] + [1]
    W = [hyper_parameters([layers[l-1], layers[l]]) for l in range(1, len(layers))]
    b = [hyper_parameters([1, layers[l]]) for l in range(1, len(layers))]
    a = [hyper_parameters_A([1, layers[l]]) for l in range(1, len(layers))]
    

    
    x_train = tf.placeholder(tf.float32, shape=[None, 1])
    y_train = tf.placeholder(tf.float32, shape=[None, 1])
    y_pred = DNN(x_train, W, b,a)
    loss = tf.reduce_mean(tf.square(y_pred - y_train)) + \
                    (1.0/(tf.reduce_mean( tf.exp(tf.reduce_mean(a[0])) + tf.exp(tf.reduce_mean(a[1]))+ tf.exp(tf.reduce_mean(a[2]))+ \
                     tf.exp(tf.reduce_mean(a[3])))))
                    
    train = tf.train.AdamOptimizer(2.0e-4).minimize(loss)
 

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    nmax = 15001
    n = 0

    MSE_hist = []
    Sol = []
    a_hist = []

    while n <= nmax: #and err > 1.0e-6:
        n = n + 1
        loss_, _, y_ = sess.run([loss, train, y_pred], feed_dict={x_train: x, y_train: y})
        err = loss_
        a_value  = sess.run(a, feed_dict={x_train: x, y_train: y})
        a_value = np.concatenate(a_value, axis=1)
        a_value = np.reshape(a_value, (1, -1))  
        a_hist.append(a_value)
        
        MSE_hist.append(err)
        
        if n == 2000 or n==8000 or n==15000: 
   
            Sol.append(y_)
            print('Steps: %d, loss: %.3e'%(n, loss_))  

    Solution = np.concatenate(Sol, axis=1)
    ###############################################################
    A_value = np.concatenate(a_hist, axis=0)

    ##################### DATA storing (.mat file) #######################
    with open('History_NN.mat','wb') as f:
     scipy.io.savemat(f, {'MSE_hist': MSE_hist})
     
    ###################### Plotting
    fig, ax = newfig(1.0, 1.1)
    plt.figure()
    plt.plot(x[0:-1], y[0:-1], 'k-',  label = 'Exact')
    plt.plot(x[0:-1], Solution[0:-1,-1], 'yx-',  label = 'Predicted at Iter = 15000')
    plt.plot(x[0:-1], Solution[0:-1,1], 'b-.',  label = 'Predicted at Iter = 8000')
    plt.plot(x[0:-1], Solution[0:-1,0], 'r--',  label = 'Predicted at Iter = 2000')
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.legend(loc='upper left')
    plt.show()
