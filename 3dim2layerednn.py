#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:09:14 2018

@author: atharvajoshi
"""

import numpy as np
x=np.array([[1,2,3],[2,4,6],[1,3,5],[2,5,6],[7,8,9],[4,6,10]])
y=np.transpose(np.array([[6,12,9,13,24,20]]))

w1=np.random.rand(3,1)
w2=np.random.rand(3,1)
w3=np.random.rand(3,1)
w4=np.random.rand(3,1)

for p in range(100000):
    #forwardpass
    u1=x.dot(w1)
    u2=x.dot(w2)
    u3=x.dot(w3)
    v=np.concatenate((u1,u2,u3),axis=1)
    u4=v.dot(w4)
    
    #backprop
    dw4=((-1)*(np.transpose(v)).dot(y-u4))
    dv=(y-u4).dot(w4.T)
    w4=w4-((0.0001)*dw4)
    dw123=(((dv).T).dot(x)).T
    dw1=dw123[:,[0]]
    dw2=dw123[:,[1]]
    dw3=dw123[:,[2]]
    w1=w1-((.0001)*dw1)
    w2=w2-((.0001)*dw2)
    w3=w3-((.0001)*dw3)
    
#predicting using trained model
def predict(t):
    r1=t.dot(w1)
    r2=t.dot(w2)
    r3=t.dot(w3)
    y1=np.concatenate((r1,r2,r3),axis=1)
    pred=y1.dot(w4)
    print (pred)

predict(np.array([[2,4,6]]))
predict(np.array([[1,4,6]]))
predict(np.array([[3,4,6]]))
predict(np.array([[2,3,6]]))





    
    

