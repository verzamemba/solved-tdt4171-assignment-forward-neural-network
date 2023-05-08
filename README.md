Download Link: https://assignmentchef.com/product/solved-tdt4171-assignment-forward-neural-network
<br>
<h1>1           Feed-forward Neural Network</h1>

In this assignment, you will implement a feed-forward neural network (NN) with one hidden layer that supports binary classification (shown in Figure 1b). The input layer should be able to take <em>n </em>(arbitrary number) features. The hidden layer should support sigmoid activation functions (see Figure 2a) and 25 hidden units. Unlike the decision tree from Assignment 4, a NN is a parametric model meaning that it learns from data by optimizing some parameters <em>w</em><sup>#» </sup>guided by a loss function    (1)

where D is set of training examples, each of the form <em>d </em>= h<em>x<sub>d</sub>,t<sub>d</sub></em>i. We will refer to this process as training. To train a feed-forward NN with one hidden layer, you will need to partially<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> implement the backpropagation algorithm outlined in Figure 18.24[1]. Be aware that Figure 18.24 has an error. Line 7 and 8: “<strong>for each </strong>weight <em>w<sub>i,j </sub></em>in <em>network </em><strong>do</strong>; <em>w<sub>i,j </sub></em>← a small random number” should come before line 6: “<strong>repeat</strong>”. Although the goal is to implement a feed-forward NN with one hidden layer, you will still get partial points if only a perceptron is implemented (see Figure 1a). We will allocate points for this assignment according to how many of the functionalities listed in Table 1 are implemented. Each of the tasks has its own unit test that can be used to assess whether the functionality is implemented correctly or not.

Table 1: Point allocation. Functionality describes what needs to be implemented to get the points. With each functionality, we have provided a unit test that you can use to assess if the functionality is correctly implemented.




Input Layer                Output Layer

<ul>

 <li>A perceptron</li>

</ul>

Input Layer                 Hidden Layer                Output Layer

<ul>

 <li>A feed-forward NN with one hidden layer and 25 hidden</li>

</ul>

units




Figure 1: The circles represent neurons, and a collection of neurons on the same level is named a layer. <em>b</em><sub>0</sub><em>,b</em><sub>1 </sub>are the bias neurons.

(a) Sigmoid function                                               (b) The derivative of the sigmoid function

<em><u>d</u></em>

<em>dx</em><em>σ</em>(<em>x</em>) = <em>σ</em>(<em>x</em>)(1 −<em>σ</em>(<em>x</em>))

Figure 2: The activation and its derivative you are going to use for this assignment.

<h2>1.1         Implementation</h2>

We will provide the skeleton code of a NN for you to implement a feed-forward NN. Please use Python 3.8 or newer (<a href="https://www.python.org/downloads/">https://www.python.org/downloads/</a><a href="https://www.python.org/downloads/">)</a>. The skeleton code contains unit tests to help you assess whether the tasks are implemented correctly or not. Please run them to see if your implementation works before delivering your code. However, note that the unit tests are not a guarantee for correctness because there can still be edge cases the unit tests do not cover.

We will now provide an overview of the skeleton code; more details can be found in the code. The constructor of the NN class takes two arguments:

<strong>input </strong><strong>dim. </strong>An integer specifying the input dimension of the NN. This is the same as the number of features in the dataset.

<strong>hidden layer. </strong>A boolean specifying whether or not to initialize the NN with a hidden layer. We have added this option, so it should be possible to get points for implementing only a perceptron.

You will need to make changes to it. For example, assigning the arguments to variables and so forth. Your task is then to implement the methods train() and predict(x: numpy.ndarray).

<strong>train. </strong>This method should implement the backpropagation algorithm outlined in Figure 18.24[1] that is used to train NNs. To run the backpropagation algorithm, some hyperparameters, such as learning rate, are needed. We have given all the necessary hyperparameters in the constructor of the NN class.

<strong>predict. </strong>This method should take an example (<em>x<sub>d</sub></em>, a NumPy vector) and classify it by outputting a prediction. The classification involves passing the example through the NN. This is known as the forward pass and involves sending the example <em>x<sub>d </sub></em>through the network from input to the output layer.

<ol>

 <li>At the hidden layer, multiply the input by the weights of the hidden layer and then add in the bias. Finally, apply the sigmoid activation function to the sums to produce the outputs. This step applies only if you implement a NN with a hidden layer (see Figure 1b).</li>

 <li>At the output layer, multiply the output of the input/hidden layer with the weights of the output layer and then add in the bias. Apply the sigmoid activation function to the sum to produce the output. The output should be a scalar value between 0 and 1.</li>

</ol>

You are free and encouraged to implement additional methods and classes to support your implementation as long as the methods train() and predict() are implemented as intended and the unit tests pass. To implement the NN, it can help to implement classes for layer and neuron.

<h3>Note</h3>

Using implementation of NNs from machine learning and deep learning libraries such as Scikitlearn[2] and Keras[3] are not allowed (fully or partially). Furthermore, it is not allowed to copy any code from the internet (using information from Piazza is allowed). However, supporting libraries, such as NumPy[4] is necessary and allowed.

Read the entire assignment, the comments in the provided skeleton code, Chapter 18[1] (parts of the syllabus) and the slideset for Lecture 8 before writing any code. The book provides a highlevel overview of the backpropagation algorithm. On the other hand, the lecture slides explicitly derives the equations you need to implement the NN. Hence, it is essential to read all of the teaching materials.

The code must be runnable without any modifications after delivery. Moreover, the code must be readable and contain comments explaining it. This is especially important if the code does not work. Finally, do not archive your source file when delivering on <em>Blackboard</em>. Failing to follow the instructions may result in points being deducted.

<h2>1.2         Datasets</h2>

We will now provide an overview of the dataset you will use to train and test the NN. You will use the breast cancer Wisconsin dataset[5] (<a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)">https://archive.ics.uci.edu/ml/datasets/Breast+ </a><a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)">Cancer+Wisconsin+(Diagnostic)</a><a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)">)</a> for this assignment. The dataset can be found on Blackboard under “Assignments” along with this assignment text. This is a binary classification dataset (see Table 2 for its content). We have preprocessed the dataset and provided the method to load the dataset loaddata(filepath: str) in the skeleton code. By default, the method assumes that the dataset is in the current working directory. However, you can change the path by providing the method with a different filepath argument, which is a string. The data will be loaded into four different variables:

<strong>self.x train </strong>A matrix (numpy.ndarray) where each row represents one example and each column a feature. You will need to use this to implement the train() method.

<strong>self.y train </strong>A vector (numpy.ndarray) of labels for the examples in self.xtrain. The vector’s length is the same as the number of rows in self.xtrain. You will need to use this to implement the train() method.

<strong>self.x test </strong>Same format as self.xtrain used in the unit tests. You do not need to touch this. <strong>self.y test </strong>Same format as self.ytrain used in the unit tests. You do not need to touch this.

More details are given in the skeleton code on how to use the dataset.

Table 2: Properties of the breast cancer Wisconsin dataset.

<h1>References</h1>

<ul>

 <li>Russell and P. Norvig. <em>Artificial Intelligence: A Modern Approach</em>. Always learning. Pearson, 2016.</li>

 <li>Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. <em>Journal of Machine Learning Research</em>, 12:2825–2830, 2011.</li>

 <li>Franc¸ois Chollet et al. Keras. <a href="https://keras.io/">https://keras.io</a><a href="https://keras.io/">,</a></li>

 <li>Charles R. Harris, K. Jarrod Millman, St’efan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fern’andez del R’ıo, Mark Wiebe, Pearu Peterson, Pierre G’erard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Array programming with NumPy. <em>Nature</em>, 585(7825):357–362, September 2020.</li>

</ul>

Dheeru Dua and Casey Graff. UCI machine learning repository, 2017

<a href="#_ftnref1" name="_ftn1">[1]</a> We say partially because Figure 18.24 generalizes the backpropagation algorithm to arbitrary NN sizes.