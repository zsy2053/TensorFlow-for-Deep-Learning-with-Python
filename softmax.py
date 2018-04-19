import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
type(mnist)
mnist.train.images
mnist.train.num_examples
mnist.test.num_examples

import matplotlib.pyplot as plt
single_image = mnist.train.images[1].reshape(28,28)
print(single_image)
plt.imshow(single_image, cmap='gist_gray')
plt.show()

#PLACEHOLDERS
x = tf.placeholder(tf.float32, shape=[None, 784])

#VARIABLES
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# CREATE graph operations
y = tf.matmul(x, W) + b

#LOSS FUNCTION
# PLACEHOLDERS
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# Session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y})

    # EVALUATE THE MODEL
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
    #[true, false, true .....] -->[1,0,1 .....]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #predicted [3,4] true [3,9]
    #[True, false]
    #[1.0, 0.0]
    #0.5
    print(sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels}))
