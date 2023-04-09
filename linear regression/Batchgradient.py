# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###calcul de l'erreur
def fonction_coût (X,y,theta): 
    
    J = (1 / (2 * X.shape[0])) * np.sum((np.dot(X,theta)-y)**2)
    return J
        
###Algo gradient (reajustement des paramètre)

def batch_gradient (X,y,theta, alpha) :
    l = np.dot(np.transpose(X),(np.dot(X,theta) - y))
    grad = (1/X.shape[0])*l
    return theta - alpha*grad

    

house_data = pd.read_csv('house.csv')

house_data = house_data[house_data['loyer'] < 10000]
#plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
#plt.show()

X = np.array([np.ones(house_data.shape[0]), house_data['surface'].values]).T
y = np.array(house_data['loyer'])
y = y.reshape(y.shape[0], 1)

alpha = 0.0001
epsilon = 0.01
e = 0
iteration = 0
E0 = 10

theta = np.random.rand(2,1)

#v = (np.dot(X, theta) - y)**2
J = fonction_coût(X, y, theta)

theta = batch_gradient (X,y,theta,alpha);

while (abs(J - e) > epsilon) :
    iteration =+ 1
    
    theta = batch_gradient (X,y,theta,alpha);
    e = J
    J = fonction_coût(X, y, theta)
    print (J-e)
    
plt.plot(X[:, 1], y, 'ro', markersize=4)

x_values = [np.min(X[:, 1]), np.max(X[:, 1])]
y_values = theta[0] + theta[1] * x_values
plt.plot(x_values, y_values, label='Regression Line')

plt.xlabel('Surface')
plt.ylabel('Loyer')
plt.title('Régression linéaire avec descente de gradient')

plt.legend()
plt.show()
    
