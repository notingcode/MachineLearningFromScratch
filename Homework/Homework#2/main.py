import random # used to set random seed
import numpy as np # used for linear algebra operations
import pandas as pd # used to load data
import matplotlib.pyplot as plt # used to plot data

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

    def rand_centroids(self, ): # set initial random centroids
        data = self.data.to_numpy() # change pandas dataframe to numpy array
        rand_idx = np.random.choice(len(data), size=self.k, replace=False) # choose k random indices of data
        sampled_cen = data[rand_idx,:] # set data that correspond to random indices as initial centroids
        return sampled_cen # return init centers

    def update_centroids(self, data, prev_centroids): # update centroids based on updated clusters
        idx = self.closest_centroid(data, prev_centroids) # get the indices of closest centroids for each point
        centroids = [data[idx == k_i].mean(axis=0) for k_i in range(self.k)] # set new centorids with the mean of points in each cluster
        centroids = np.stack(centroids, axis=1) # stack each centroid as numpy array so that it is suitable for next update
        is_changed = self.update_check(centroids, prev_centroids) # check if the coordinates of centroids changed
        return centroids, is_changed # return centroids and their update status

    def update_check(self, centroids, prev_centro): # check if coordinates of centroids changed
        return np.array_equal(centroids, prev_centro) # checks for differences in size and elements of two numpy arrays

    def train(self, ): # train the model with data for a given iteration
        data = self.data.to_numpy() # change pandas dataframe to numpy array
        data = np.expand_dims(data, axis=1) # expand dimension to allow broadcasting between data and centroids
        centroids = self.rand_centroids() # set initial centroids
        for i in range(self.iteration): # update each iteration
            centroids, update_status = self.update_centroids(data, centroids) # get new centroids and its update status
            if(update_status == True): # break if there is no change in centroids
                break
        idx = self.closest_centroid(data, centroids) # get the indices of closest centroid for each data point
        centroids = centroids.reshape(-1, 2) # reshape centroids to proper size
        return idx, centroids # return result

    def closest_centroid(self, data, centroids): # get the indices of closest centroids
        distance = np.linalg.norm(data-centroids, axis=2) # get the euclidean distances between each data point and all the centroids
        idx = np.argmin(distance, axis=1) # find the index of the closeset centroid for each point
        return idx # return a horizontal vector with indices of closest centroid to each data point 

if __name__ == '__main__': # Start from main
    data = pd.read_csv("data.csv") # load data
    model1 = kmeans_(k=5, data=data) # implemented model init setting
    idx, centroids = model1.train()
    plt.scatter(data['Sepal width'], data['Sepal length'], c=idx, cmap='Set1', alpha=0.5) # plt scatter for each clusters
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("implementaion") # set title
    plt.show() # show plot
  
    model2 = KMeans(n_clusters=5, init='random', random_state=seed_num, max_iter=400).fit(data) # sklearn model init setting
    predict = pd.DataFrame(model2.predict(data)) # update predict label
    predict.columns = ["predict"] # Set col name
    data = pd.concat([data, predict],axis=1) # concat data
    predict.columns=['predict'] # Set col name
    plt.scatter(data['Sepal width'], data['Sepal length'], c=data['predict'], cmap='Set1',alpha=0.5) # scatter plot
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("from scikit-learn library") # set title
    plt.show() # show plot