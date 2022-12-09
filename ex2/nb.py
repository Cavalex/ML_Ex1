import numpy as np

class GaussianNaiveBayes:

    def fit(self, X, y):        
        self.unique_labels = np.unique(y)

        self.params = []
        for label in self.unique_labels:
            label_features = X[y == label]
            self.params.append([(col.mean(), col.var()) for col in label_features.T])


    """ def fit(self, X, y):
        # get number of samples (rows) and features (columns)
        self.n_samples, self.n_features = X.shape
        # get number of uniques classes
        self.n_classes = len(np.unique(y))
        
        # create three zero-matrices to store summary stats & prior
        self.mean = np.zeros((self.n_classes, self.n_features))
        self.variance = np.zeros((self.n_classes, self.n_features))
        self.priors = np.zeros(self.n_classes)

        for c in range(self.n_classes):
            # create a subset of data for the specific class 'c'
            X_c = X[y == c]
            
            # calculate statistics and update zero-matrices, rows=classes, cols=features
            self.mean[c, :] = np.mean(X_c, axis=0)
            self.variance[c, :] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / self.n_samples """






"""     
    features: np.ndarray # columns of the input
    labels: np.ndarray # output

    # Fits the model
    def fit(self) -> None:        
        self.unique_labels = np.unique(self.labels)

        self.params = []
        for label in self.unique_labels:
            label_features = self.features[self.labels == label]
            self.params.append([(col.mean(), col.var()) for col in label_features.T])


    # Performs the inference using bayes theorem: P(A|B) = P(B|A) * P(A) / P(B)
    def predict(self, features: np.ndarray):# -> np.ndarray:
        num_samples, _ = features.shape
        predictions = np.empty(num_samples) # initializes it

        for feature in features:
            posteriors = []
            for feature in features:
                posteriors = []
                for label_idx, label in enumerate(self.unique_labels):

                    prior = (self.labels==label).mean()


 """