####################################################################################
# File Name :                                                                      #  
# Date  : 2020/00/00                                                               #  
# OS : Windows 10                                                                  #  
# Author :                                                                         #
# -------------------------------------------------------------------------------  #  
# requirements : python 3.x                                                        #
#                                                                                  #
####################################################################################   

import random                      
import numpy as np                 
import pandas as pd                 
import matplotlib.pyplot as plt    
import warnings                     

try:
    from sklearn.cluster import KMeans  # check installation of sklearn
except:
    print("Not installed scikit-learn.") 
    pass


if __name__ == '__main__': # Start from main
    #Implement Load and Plot Code here
    data = pd.read_csv("data.csv")

    plt.scatter(data['Sepal width'], data['Sepal length'])
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Sepal Length (cm)")
    plt.show() # show plot