'''
Tensorflow implementation of network3.py
Not GPU accelrated 
Network Architecture:

CNN:

1st Convolutional Layer:
Convolution computes 32 features for each 5x5 patch
Max Pooling 

2nd Convolution Layer:
64 features for each 5x5 patch
Max Pooling

Image reduced to 7x7, add fully connected layer of 1024 (1K) ReLU 

Dropout before readout layer

Readout layer, Softmax regression for outputs 

'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

print("retrieving data...")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print("finishing retrieving data, starting interactive session...")
# equivalent to ipynb 
sess = tf.InteractiveSession()

# tf uses a "computational graph" to optimize things like matmul
# Builds an external computation graph and optimizes it

print("initializes interactive session, building variables...")

# Placeholders (like passing a variable via reference in C++)

# 784 = Flattened pixel image, None = batch size (can be any size)
# shape arg = optional, better readability
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# Weights W, Biases b live in tf's computational graph as Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Initialize Variable with zeros of shape that we need
# To use variables in the session, we must initialize them in that session

sess.run(tf.global_variables_initializer())

## GLOBAL FUNCTIONS FOR CNN ##

# init weights w small amnt of noise for symmetry breaking and avoiding 0 gradients
# good practice  = slightly positive initial bias (avoid dead neurons)
# ReLU 
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution options: Stride length, how to handle padding?
# Stride 1, zero padded so output.size() == input.size()
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# regular mac pool over 2x2 blocks
def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1],
        strides=[1,2,2,1], padding='SAME')



class SoftmaxRegression:
    def run(self):

        # regression model
        y = tf.matmul(x, W) + b
        
        # loss function, part 1
        # Cross Entropy / Softmax

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        # softmax_cross... = internally applies softmax on model, sums across all classes
        # Reduce mean takes average over these sums

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        # Step length = .5, descend the cross entropy
        # train_step adds operations to computational graph (computer gradients, compute param update
        # steps, apply updates)

        for i  in range(1000):
            # load 100 training examples per iteration
            batch = mnist.train.next_batch(100)

            # put in x, get out y_ (replace placeholder) based on predefined train_step
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        # tf.argmax = index of highest entry of any tensor along any axis
        # ie. tf.argmax(y, 1) = label most likely

        # list of bools
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        # calculate, print accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("Accuracy is {:.2%}" .format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

class DeepConvolutionNetwork:
     def run(self):
        # 1st conv. layer
        # 32 features for each 5x5 patch
        # first two dimensions = patch size, 3rd=num input channels, last =# output channels
        # bias vector w component for each output channel

        W_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])

        # to apply later, reshape x to 4d tensor
        # 2nd,3rd dimension = image w, height
        # final dimension = num color channels

        x_img = tf.reshape(x, [-1, 28, 28, 1])

        # convolve x_img w weight tensor, add bias,
        # Apply relu, finally maxpool
        # max_pool_2x2 -> 14x14 output image

        h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # 2nd later
        # computes 64 features for each 5x5 patch
        # max pool 2x2 -> 7x7 output image
        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Fully connected layer w 1K (1024) neurons
        # Reshape tensor (h_pool2) into batch of vectors, multipy by weight matrix, add bias, apply relu

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # apply dropout to reduce overfitting before readout
        # placeholder for probability that a neurons output is kept during output
        # (allows us to turn on dropout for training, off during testing)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Readout layer
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Replace steepest gradient descent w ADAM optimizer
        # Include "keep_prob" in feed_dict to control dropout rate
        # Logging to every 100th iteration
        
        # Loss function
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        
        # Train with ADAM optimizer
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # Set correct_prediction as evaluation fun
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # initialize all the variables that we declared here
        sess.run(tf.global_variables_initializer())
        
        # 20000 batches
        for i in range(20000):
          # batch sz = 50
          batch = mnist.train.next_batch(50)
          
          # logging
          if i%100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0
                })

            print("step %d, training accuracy %g" % (i, train_accuracy))
          
          # run with 50% dropout rate
          train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        # output
        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
    softmax_regression = SoftmaxRegression()
    print("initialized softmax regression instance, starting now...")
    softmax_regression.run()

    convolution_instance = DeepConvolutionNetwork()
    print("initialized convolution NN instance, starting now...")
    convolution_instance.run()