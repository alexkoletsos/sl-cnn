Alexandra Koletsos
ak4749

1. A one-hot encoding is a method of converting categorical data into feature 
vectors that can be used for machine learning models. Essentially, a one-hot 
encoding creates a binary vector for each category of data and assigns a value 
of 1 at the index that corresponds to the associated category for each vector 
(and 0s at all other indices). It is extremely useful when dealing with data 
that is independent from each other and when dealing with categorical data in 
general, which the model can then use for classification. A one-hot encoding is 
implemented in keras using the to_categorical() method.

2. Dropout removes random nodes from the input and hidden layers based on a 
given probability. For example, if the dropout probability given for a hidden 
layer is 0.2, then 20% of nodes in that layer will be removed. This helps with 
overfitting because it prevents neurons from co-adapting (overly relying on 
one another) and thus improves generalization.

3. The sigmoid activation function “squashes” a value to range between [0,1] 
while ReLU (denoted by R(x)=max(0,x)) sets the value to 0 if it is less than 0 
and keeps it the same otherwise. ReLU does not suffer from the gradient descent 
problem and is more computationally efficient when compared to the sigmoid 
activation function.

4. The softmax function is necessary in the output layer because it outputs a 
probability distribution over the possible classes (since it ranges between 0 
and 1), which the CNN uses to make a prediction about the class of an input 
image.

5. Convolution Output dimension = [(I - F + 2 * P) / S] + 1 x D.
Then, we have (100 - 5 + 2 * 0) + 1 = 96, so the dimensions of the convolution 
outputs would be 96x96x16 (since there are 16 filters). The dimensions of 
the max pooling layers would be [(I - F) / S] + 1 x D, which gives us the 
dimensions 48x48x16.

A brief description of my model.
I used two convolutional blocks that each contain two layers, implementing
128 filters on the layers of the first block and 256 on the layers of the second
block. After doing some research, I learned that the number of filters used
is typically powers of 2 (by convention!). I played around with numbers and 
these seemed to make my code the most accurate. Another convention I learned
was that filters tend to double between convolutional blocks, hence why I 
chose 128 and 256. This allows the model to detect more combinations of complex
patterns as it moves through the network. The dimensions of the kernel (3x3) 
and maxpool (2x2) I chose are the ones most commonly used (which I read about at 
https://www.quora.com/Are-maxpooling-layer-kernel-sizes-in-CNNs-generally-smaller-than-convolutional-layer-kernel-sizes-Why).
It was difficult for me to figure out exactly what dimensions worked and why,
as I ran into the error "Negative dimension size caused by subtracting 3 from 2"
for example... This was caused because I tried adding a third convolutional 
block, but the input tensor became too small for my kernel size. So, I 
decided to reduce the number of blocks/layers and increase the filter size
instead. A dropout of 0.5 was added for each block to prevent overfitting.
I used Flatten() to create the fully connected layer and then applied
ReLU and softmax to the fully connected layer for reasons discussed in (5).
The normalization techniques were inspired by Keras.io, found at the link
https://keras.io/examples/vision/mnist_convnet/ (specifically dividing by
255 and using the to_categorical() method). Lastly, I used a test size of 0.2 
because online resources recommended an 80/20 split as it allows for adequate 
testing data while still preventing overfitting.