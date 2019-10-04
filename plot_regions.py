import matplotlib
matplotlib.rcParams["backend"] = "Agg"


'''This plotting method is refrenced from 
    Python Machine Learning Book
    by Sebastian Raschka & Vahid Mirjalili
'''
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


#matplotlib.use('TkAgg')
def plot_decision_regions(X, y, classifier, resolution=0.02):
     # setup marker generator and color map
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))]) #Set colors for each unique target value
    
    
    #get min and max values from each row
    x1_min, x1_max = X[:, 0].min() -1, X[:,0].max() +1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() +1
    
    
    matrix1, matrix2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([matrix1.ravel(), matrix2.ravel()]).T)
    Z = Z.reshape(matrix1.shape) #Reshapes to matrix1 size
    
    plt.contourf(matrix1, matrix2, Z, alpha=0.4, cmap=cmap) #Draws Line. Note: xx1,xx2, and Z must be same .shape
    plt.xlim(matrix1.min(), matrix1.max())
    plt.ylim(matrix2.min(), matrix2.max())
    for idx, cl in enumerate(np.unique(y)): #Graphs all points that have -1 and 1
        plt.scatter(X[y == cl, 0],  X[y == cl, 1],   alpha=0.6,   c=cmap(idx)    ,edgecolor='black')



plt.show(block = True)
