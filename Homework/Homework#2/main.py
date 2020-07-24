import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import KMeans  # check installation of sklearn
except:
    print("Not installed scikit-learn.")  # print general error  
    pass

seed_num = 777  # set random seed
np.random.seed(seed_num) # seed setting

class kmeans_:
    def __init__(self, k, data, iteration=400): # initalize
        self.k = k # number of cluster
        self.data = data # data
        self.iteration = iteration # set iteration to 400

    def rand_centroids(self, ): # Set initial random centroids
        data = self.data.to_numpy()
        rand_idx = np.random.choice(data.shape[0], size=self.k, replace=False)
        sampled_cen = data[rand_idx,:]
        return sampled_cen # return init centers

    def update_centroids(self, data, prev_centroids): # update centroids based on updated clusters
        idx = self.closest_centroid(data, prev_centroids)
        centroids = [data[idx == k_i].mean(axis=0) for k_i in range(self.k)]
        centroids = np.stack(centroids, axis=1)
        is_changed = self.update_check(centroids, prev_centroids)
        return centroids, is_changed # return centroids and their update status

    def update_check(self, centroids, prev_centro): # check if coordinates of centroids changed
        return np.array_equal(centroids, prev_centro)

    def train(self, ): # train the model with data for a given iteration
        data = self.data.to_numpy()
        data = np.expand_dims(data, axis=1)
        centroids = self.rand_centroids()
        iteration = self.iteration
        for i in range(iteration):
            centroids, update_status = self.update_centroids(data, centroids)
            if(update_status == True):
                break
        idx = self.closest_centroid(data, centroids)
        centroids = centroids.reshape(-1, 2)
        return idx, centroids # return result

    def closest_centroid(self, data, centroids): # Get the indices of closest centroids
        distance = np.linalg.norm(data-centroids, axis=2)
        idx = np.argmin(distance, axis=1)
        return idx # return a horizontal vector with indices of closest centroid to each data point 

if __name__ == '__main__': # Start from main
    data = pd.read_csv('data.csv') # load data
    model1 = kmeans_(k=3, data=data) # implemented model init setting
    idx, centroids = model1.train()
    plt.scatter(data['Sepal width'], data['Sepal length'], c=idx) # plt scatter for each clusters
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("implementaion") # set title
    plt.show() # show plot

    model2 = KMeans(n_clusters=3, init='random', random_state=seed_num, max_iter=400).fit(data) # sklearn model init setting
    predict = pd.DataFrame(model2.predict(data)) # update predict label
    predict.columns = ["predict"] # Set col name
    data = pd.concat([data, predict],axis=1) # concat data
    predict.columns=['predict'] # Set col name
    plt.scatter(data['Sepal width'], data['Sepal length'],c=data['predict'], alpha=0.5) # scatter plot
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("from scikit-learn library") # set title
    plt.show() # show plot