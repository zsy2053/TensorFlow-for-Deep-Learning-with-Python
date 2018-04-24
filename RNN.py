import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class TimeSeriesData():
    def __init__(self, num_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin,xmax,num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps,return_batch_ts=False):
        # get a random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)
        # convert to be on time series
        ts_start=rand_start * (self.xmax - self.xmin - (steps*self.resolution))
        # create batch time series on the x axis
        batch_ts = ts_start + np.arange(0.0, steps+1)*self.resolution
        # create the y data for the time series x axis from previous steps
        y_batch=np.sin(batch_ts)
        # formatting for rnn
        if return_batch_ts:
            return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1), batch_ts
        else:
            return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1)

ts_data = TimeSeriesData(250,0,10)
# plt.plot(ts_data.x_data, ts_data.y_true)
num_time_steps = 30
y1,y2,ts = ts_data.next_batch(1,num_time_steps,True)
# plt.plot(ts.flatten()[1:], y2.flatten(), '*')
# plt.plot(ts_data.x_data,ts_data.y_true,label='Sin(t)')
# plt.plot(ts.flatten()[1:],y2.flatten(),'*',label="Single Training instace")
# plt.legend()
# plt.tight_layout()

# TRAINING Data
train_inst = np.linspace(5, 5 + ts_data.resolution * (num_time_steps + 1), num_time_steps + 1)
print(train_inst)
plt.title('A TRAINING INSTANCE')
plt.plot(train_inst[:-1],ts_data.ret_true(train_inst[:-1]),'bo',markersize=15,alpha=0.5,label='INSTANCE')
plt.plot(train_inst[1:],ts_data.ret_true(train_inst[1:]),'ko',markersize=7,label='TARGET')
plt.legend()

# Creating the Model
tf.reset_default_graph()
num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.0001
num_train_iterations = 2000
batch_size = 1

# PLACEHOLDERS
X = tf.placeholder(tf.float32,[None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32,[None, num_time_steps, num_outputs])

# RNN CELL LAYER
cell1 = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons,activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell1,output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# MSE
loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

# SESSION
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
saver = tf.train.Saver()
with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for iteration in range(num_train_iterations):
        X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
        sess.run(train, feed_dict = {X:X_batch, y:y_batch})

        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            print(iteration, "\tMSE", mse)
    saver.save(sess,"./rnn_time_series_model_codealong")

with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_codealong")
    X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
    y_pred = sess.run(outputs, feed_dict={X:X_new})

plt.title("TESTING the model")
# training instance
plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), "bo", markersize=15,alpha=0.5,label='TRAINING INST')
# target to predict
plt.plot(train_inst[1:],np.sin(train_inst[1:]), "ko", markersize=15,alpha=0.5,label='TARGET')
# models prediction
plt.plot(train_inst[1:],y_pred[0,:,0],'r.',markersize=10,label='PREDICTION')

plt.xlabel('TIME')
plt.legend()
plt.tight_layout()

with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_codealong")
    #  seed zeros
    zeros_sq_seed = [0.0 for i in range(num_time_steps)]
    for iteration in range(len(ts_data.x_data)-num_time_steps):
        X_batch = np.array(zeros_sq_seed[-num_time_steps:]).reshape(1,num_time_steps,1)
        y_pred = sess.run(outputs, feed_dict={X:X_batch})
        zeros_sq_seed.append(y_pred[0,-1,0])

plt.plot(ts_data.x_data, zeros_sq_seed, 'b-')
plt.plot(ts_data.x_data[:num_time_steps],zeros_sq_seed[:num_time_steps],'r',linewidth=3)
plt.xlabel('TIME')
plt.ylabel("Y")

with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_codealong")
    #  seed zeros
    training_instance = list(ts_data.y_true[:30])
    for iteration in range(len(training_instance)-num_time_steps):
        X_batch = np.array(training_instance[-num_time_steps:]).reshape(1,num_time_steps,1)
        y_pred = sess.run(outputs, feed_dict={X:X_batch})
        training_instance.append(y_pred[0,-1,0])
plt.plot(ts_data.x_data, ts_data.y_true, 'b-')
plt.plot(ts_data.x_data[:num_time_steps],training_instance[:num_time_steps],'r',linewidth=3)
plt.xlabel('TIME')
plt.ylabel("Y")
plt.show()

# part 1
# CONSTANTS
num_inputs = 2
num_neurons = 3

# PLACEHOLDERS
x0 = tf.placeholder(tf.float32, [None, num_inputs])
x1 = tf.placeholder(tf.float32, [None, num_inputs])

# VARIABLES
Wx = tf.Variable(tf.random_normal(shape=[num_inputs, num_neurons]))
Wy = tf.Variable(tf.random_normal(shape=[num_neurons, num_neurons]))
b = tf.Variable(tf.zeros([1, num_neurons]))

# GRAPHS
y0 = tf.tanh(tf.matmul(x0,Wx) + b)
y1 = tf.tanh(tf.matmul(y0,Wy) + tf.matmul(x1,Wx) + b)

init = tf.global_variables_initializer()

# CREATE DATA
x0_batch = np.array([[0,1], [2,3], [4,5]])

# TIMESTAMP 1
x1_batch = np.array([[100,101], [102,103], [104,105]])

with tf.Session() as sess:
    sess.run(init)
    y0_output_vals, y1_output_vals = sess.run([y0,y1],feed_dict={x0:x0_batch,x1:x1_batch})

print(y0_output_vals)
print(y1_output_vals)
