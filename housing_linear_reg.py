# Program for applying linear regression on housing dataset. Here housing data is used which is available in housing_data.csv file
# To run this program keep  housing_data.csv file and this program in same folder.
# About housing_data: 
   # dataset characterisitics:multivariate
   #Associate task:          regression
   #Number of instance:      506
   #number of attribut:      14

   #Attribute Information:

  #1. CRIM: per capita crime rate by town
  #2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
  #3. INDUS: proportion of non-retail business acres per town
  #4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
  #5. NOX: nitric oxides concentration (parts per 10 million)
  #6. RM: average number of rooms per dwelling
  #7. AGE: proportion of owner-occupied units built prior to 1940
  #8. DIS: weighted distances to five Boston employment centres
  #9. RAD: index of accessibility to radial highways
  #10. TAX: full-value property-tax rate per $10,000
  #11. PTRATIO: pupil-teacher ratio by town
  #12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
  #13. LSTAT: % lower status of the population
  #14. MEDV: Median value of owner-occupied homes in $1000's

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size=50

filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("housing_data.csv"),
    shuffle=True)

# Each file will have a header, we skip it and give defaults and type information
# for each column below.
line_reader = tf.TextLineReader(skip_header_lines=1)

_, csv_row = line_reader.read(filename_queue)

# Type information and column names based on the decoded CSV.
record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0],[0.0],[0.0],[0.0],[0.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0]]

CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV = tf.decode_csv(
	                        csv_row, record_defaults=record_defaults)

# Turn the features back into a tensor.
features = tf.pack([CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV])



#variable which we need to fill when we are ready to comput the graph
x=tf.placeholder(dtype=features.dtype)
y=tf.placeholder(dtype=MEDV.dtype)

# %% We will try to optimize min_(W,b) ||(X*w + b) - y||^2
# The `Variable()` constructor requires an initial value for the variable,
# which can be a `Tensor` of any type and shape. The initial value defines the
# type and shape of the variable. After construction, the type and shape of
# the variable are fixed. The value can be changed using one of the assign
# methods.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
y_pred = tf.add(tf.mul(x, W), b)


#loss function will measure the distance between our observations and predictions
#and average over them.Here housing data have 506 instances so divide by 506.
error=tf.reduce_sum((y-y_pred)**2/506)

# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

#create the session to use the graph
with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them 
    #tf.initialize_all_variables().run()
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    x_data =np.array(sess.run(features))


    y_data =np.array(sess.run(MEDV))

    

        #gradient descent loop for 10 iteration
    for _ in range(10):
        sess.run([features,MEDV])
 
        _,loss_val=sess.run([optimizer,error],feed_dict={x:x_data,y:y_data})

        print _,loss_val

   
    coord.request_stop()
    coord.join(threads)

