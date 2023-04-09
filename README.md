# Régression linéaire avec descente de gradient

Ce code Python effectue une régression linéaire à l'aide d'un algorithme de descente de gradient sur un ensemble de données immobilières. Il utilise une librairie de calcul scientifique Numpy, une librairie de visualisation Matplotlib, et une librairie de manipulation de données Pandas.

## Prérequis

Avant d'exécuter le code, il est nécessaire d'installer les librairies suivantes :
- Numpy
- Pandas
- Matplotlib

## Utilisation

- Clonez ce repository sur votre ordinateur.
- Placez le fichier de données house.csv dans le même répertoire que le fichier de code regression_lineaire.py.
- Ouvrez un terminal ou une ligne de commande à partir du répertoire contenant les fichiers, et exécutez la commande suivante :
``` 
python regression_lineaire.py
```
Le résultat de la régression linéaire est affiché, avec une visualisation graphique de la droite de régression.

## Description du code

Le code comporte deux fonctions principales :

- `fonction_cout` : Cette fonction calcule le coût (ou l'erreur) de la régression linéaire pour un ensemble de données, à partir de la formule suivante : `J = (1 / (2 * m)) * sum((X.dot(theta)-y)**2)`, où X est la matrice de données, y est le vecteur de résultats, theta est le vecteur de coefficients de régression, et m est le nombre d'échantillons.
- `batch_gradient` : Cette fonction implémente l'algorithme de descente de gradient pour ajuster les coefficients de régression. Elle utilise la formule suivante pour calculer le nouveau vecteur de coefficients : `theta = theta - alpha * (1/m) * np.transpose(X).dot(X.dot(theta)-y)`, où alpha est le taux d'apprentissage.

Le code utilise également une boucle while pour itérer sur l'algorithme de descente de gradient jusqu'à ce que le coût converge.

## Auteur

Ce code a été écrit par [Vincent JOUANNEAU].

## License

Ce code est sous license MIT. Voir le fichier LICENSE.md pour plus de détails.
