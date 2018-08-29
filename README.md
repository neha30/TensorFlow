# Applying classification and clustering on different datasets using TensorFlow. 

## What is TensorFlow?
“TensorFlow is an open source software library for numerical computation using dataflow graphs. Nodes in the graph represents mathematical operations, while graph edges represent multi-dimensional data arrays (aka tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API.”
(https://www.tensorflow.org/)

If you know about the Numpy library in Python, then it is easy to understand TensorFlow also. A main difference between Numpy and TenorFlow is that TensorFlow follow lazy programming paradigm. It first build the graph of all operation to be done, and then when a "session" is called, it "run" the graph.

The useful way of running a program in TensorFlow is as follow:

1) Build a Computaion graph: This can be any mathematical operation TensorFlow support.
2) Initialize variable: To compile variable define previously.
3) Create session
4) Run graph in session: The compiled graph is passed to session, which start its execution.
5) Close session: shutdown the session

---> Basic setup:
     Install TensorFlow  (https://www.tensorflow.org/install/)
	 
---> Execution:
     $ python <filename>

