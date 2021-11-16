from sklearn.decomposition import PCA

# Get 2 main pca directions of the models parameters
def get_pca_directions(models):
    params = []
