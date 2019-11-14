class LinearRegressor:
    def add_bias(self, X):
        return np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    
    def fit(self, X, y, lr=0.001, iter_= 1000, thres=0.0001):
        self.weights = np.zeros(X.shape[1] + 1)
        X = self.add_bias(X) # add bias 1 to first column
        while True:
            gradient = np.dot((y-self.predict(X, False)), X)
            update = lr * gradient
            self.weights = self.weights + update
            if np.max(np.absolute(update)) < thres: break
                
    
                
    def predict(self, X, no_bias=True):
        if no_bias: X = self.add_bias(X)
        return np.dot(X, self.weights)
