from sklearn import datasets, metrics, preprocessing
import tensorflow.contrib.learn as skflow

#load dataset
boston = datasets.load_boston()

#scale data to 0 mean and unit standard deviation
X = preprocessing.StandardScaler().fit_transform(boston.data)
regressor = skflow.TensorFlowLinearRegressor()
regressor.fit(X, boston.target)

#predict and score
score = metrics.mean_squared_error(regressor.predict(X), boston.target)
print ("MSE: %f" % score)


