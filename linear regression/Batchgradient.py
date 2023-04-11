# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#--Calcul de la fonction coût--#
def fonction_coût (X,y,theta): 
    
    J = (1 / (2 * X.shape[0])) * np.sum((np.dot(X,theta)-y)**2)
    return J
        
#--Algo gradient (reajustement des paramètres)--#

def batch_gradient (X,y,theta, alpha) :
     
    l = np.dot(np.transpose(X),(np.dot(X,theta) - y))
    grad = (1/X.shape[0])*l
    return theta - alpha*grad

#--Coefficient de determination(pertinence du modèle)--#
def coef_determination (y, prev):
    u = ((y - prev)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1-(u/v)
    
    
#---Importation du fichier Excel---#
house_data = pd.read_csv('house.csv')

house_data = house_data[house_data['loyer'] < 10000]
#plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
#plt.show()

#---Intialisation---#

X = np.array([np.ones(house_data.shape[0]), house_data['surface'].values]).T
y = np.array(house_data['loyer'])
y = y.reshape(y.shape[0], 1)

alpha = 0.0001
epsilon = 1
e = 0
iteration = 0


#Paramètres générés aleatoirement 
theta = np.random.rand(2,1)

#Calcul à l'iteration 0 de la fonction coût
J = fonction_coût(X, y, theta)

#courbe d'apprentissage
cost_history= []
cost_history.append(J)

#---Demarage de l'algo: on cherche à faire converger la fonction coût vers sont minimun---#
while (abs(J - e) > epsilon) :
    iteration += 1
    
    theta = batch_gradient (X,y,theta,alpha);
    
    e = J
    J = fonction_coût(X, y, theta)
    cost_history.append(J)
    #print (J-e)


yprev = X.dot(theta)
print(coef_determination(y, yprev))

#---Affichage---#
#Prevision du modèle
plt.scatter(X[:,1], y)
plt.plot(X[:,1], yprev, c='r', label='Regression Line')
plt.xlabel('Surface')
plt.ylabel('Loyer')
plt.title('Régression linéaire avec descente de gradient')
plt.legend()
plt.show()

#Courbe d'apprentissage du modèle
plt.plot(range(iteration+1), cost_history, label='Courbe d\'apprentissage')
plt.xlabel('Iteration')
plt.ylabel('L\'érreur')
plt.title('Courbe d\'apprentissage du model')
plt.show()
