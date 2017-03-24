#https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html (To understand softmax classification)

import sklearn
from sklearn import datasets
from sklearn import cross_validation
import tensorflow as tf

iris = datasets.load_iris()

# Convert targets to one hot
targets_onehot = [0] * len(iris.target)
for i,target in enumerate(iris.target):
       targets_onehot[i] = [0] * 3
       targets_onehot[i][target] = 1

#data_train->training data, target_train->labels corresponding to training data
#data_test->testing data,target_test->labels corresponding to test data
data_train, data_test, target_train, target_test = cross_validation.train_test_split(iris.data, targets_onehot)

print "Training: {}\tTesting: {}".format(len(data_train), len(data_test))

#four input
tf_in = tf.placeholder("float", [None, 4])

# Weight and bias variables
#tf.zeros([4,3])->create a matrix of size 4*3 with zero entries
tf_weight = tf.Variable(tf.zeros([4,3]))
tf_bias = tf.Variable(tf.zeros([3]))

# Output
tf_softmax = tf.nn.softmax(tf.matmul(tf_in,tf_weight) + tf_bias)

# Training via backpropagation, cross_entropy is the cost function
tf_softmax_correct = tf.placeholder("float", [None,3])
tf_cross_entropy = -tf.reduce_sum(tf_softmax_correct*tf.log(tf_softmax))

# Train using tf.train.GradientDescentOptimizer
tf_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(tf_cross_entropy)

# Add accuracy checking nodes
tf_correct_prediction = tf.equal(tf.argmax(tf_softmax,1), tf.argmax(tf_softmax_correct,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

# Initialize and run
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Run the training
sess.run(tf_train_step, feed_dict={tf_in: data_train, tf_softmax_correct: target_train})
# Print accuracy
print sess.run(tf_accuracy, feed_dict={tf_in: data_test, tf_softmax_correct: target_test})
