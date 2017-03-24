#KNN on iris dataset

import sklearn
from sklearn import datasets
from sklearn import cross_validation
import numpy as np
import tensorflow as tf

iris=datasets.load_iris()
data_train, data_test, target_train, target_test = cross_validation.train_test_split(iris.data, iris.target)

#print "training data:",data_train
#print "labels of training data",target_train

print "data_train:",len(data_train)
print "data_test:",len(data_test)


xtr=tf.placeholder("float",[None,4])
xte=tf.placeholder("float",[4])

distance=tf.reduce_sum(tf.abs(tf.add(xtr,tf.neg(xte))),reduction_indices=1)

pred=tf.arg_min(distance,0)

accuracy=0

init=tf.initialize_all_variables()

#launch the graph
with tf.Session() as sess:
     sess.run(init)

     #loop over test data
     for i in range(len(data_test)):
         #get nearest neighbors
         nn_index=sess.run(pred,feed_dict={xtr:data_train,xte:data_test[i,:]})
         #get nearest neighbor class label and compare it to its true label
         print "Test",i,"prediction:",np.argmax(data_train[nn_index]),"True class:",np.argmax(data_test[i])
         #calculate accuracy
         if np.argmax(data_train[nn_index])==np.argmax(data_test[i]):
            accuracy+=1./len(data_test)
     print "accuracy is:",accuracy
