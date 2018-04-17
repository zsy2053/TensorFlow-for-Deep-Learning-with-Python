import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    epochs = 100
    for i in range(epochs):
        sess.run(train)
    final_slope, final_intercept = sess.run([m,b])
    saver.save(sess, 'models/my_first_model.ckpt')
    saver.save(sess, './my_second_model.ckpt')
    
