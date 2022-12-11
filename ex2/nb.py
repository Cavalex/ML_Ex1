import numpy as np

class GaussianNaiveBayes:

    """
    P(A | B) = P(B | A) * P(A) / P(B)
    =
    P(y | X) = P(X | y) * P(y) / P(X)

    P(X) -> MARGINAL LIKELYHOOD
    P(y) -> PRIOR -> FREQUENCY OF EACH CLASS
    P(X / xi | y) -> LIKELIHOOD OR CLASS CONDITIONAL PROBABILITY-> MODEL WITH GAUSSIAN
    P(y | X) -> POSTERIOR

    X -> FEATURES
    y -> LABELS

    TRAINING: 
        MEAN, VARIANCE AND PRIOR OF EACH CLASS

    PREDICTIONS: 
        POSTERIOR OF EACH CLASS
            CALCULATE THE LIKELIHOOD
        CHOOSE CLASS WITH HIGHEST POSTERIOR PROBABILITY
    """

    # training the model, i.e. calculating everything necessary to get the posterior on the predict function.
    def fit(self, X, y):
        self.labels = y
        self.unique_labels = np.unique(y)

        # we need the means and variances to calculate the likelihood
        self.means = []
        self.vars = []
        self.priors = []

        # for every label, calculate the mean and variance of all features
        for label in self.unique_labels:
            label_features = X[y == label] # remove all other classes except the one selected
            self.means.append([col.mean() for col in label_features.T])
            self.vars.append([col.var() for col in label_features.T])
            self.priors.append(label_features.shape[0] / X.shape[0]) # number of samples of this class divided by total number of samples

        #print("\nX:", X)
        #print("\nvars:", self.vars)
        #print("\nmeans:", self.means)
        #print("\npriors:", self.priors)

    # predicting the target values given X features. will basically calculate the posterior of each class given the training data
    def predict(self, X):

        num_samples, num_features = X.shape

        predictions = []

        # we now calculate the posterior for each label
        # we will use the formula on notes/posterior_formula.png
        for x in X:
            posteriors = [] # we want to select the most likely one according to each label/class
            for i, c in enumerate(self.unique_labels):
                likelihoods = []
                for j, value in enumerate(x):

                    #print("\nx:", x)
                    #print("\nvars:", self.vars)
                    #print("\nmeans:", self.means)
                    #print("\npriors:", self.priors)

                    #print("\nfeature:", x)
                    #print("\nmeans:", self.means[i])
                    #print("\nvars:", self.vars[i])
                    #break

                    # now we calculate the likelihood of each feature and append it to then sum everything together
                    # this is like P(xi | y)
                    # so we need to sum to get P(X | y)
                    likelihood = np.log(self.likelihood(value, self.means[i][j], self.vars[i][j]))
                    likelihoods.append(likelihood)

                posterior = np.sum(likelihoods) + self.priors[i]
                posteriors.append(posterior)

            #print(posteriors)
            predictions.append(np.argmax(posteriors))

        return np.array(predictions) # it must be a np array to work with scikit later on

    # calculates the gaussian likelihood of the data with the given mean and variance.
    def likelihood(self, x, mean, var):

        # NOTE: Added in denominator to prevent division by zero
        eps = 1e-4

        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2 / (2 * var + eps)))

        return coeff * exponent

# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))
