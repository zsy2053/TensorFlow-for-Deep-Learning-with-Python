from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
from tensorflow import estimator
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import losses, optimizers, metrics, activations
from tensorflow.contrib.layers import fully_connected
from sklearn.metrics import confusion_matrix, classification_report

wine_data = load_wine()
type(wine_data)
print(wine_data.keys())
print(wine_data['DESCR'])
feat_data = wine_data['data']
labels = wine_data['target']

X_train, X_test, y_train, y_test = train_test_split(feat_data, labels, test_size=0.3, random_state=101)
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

X_train.shape
feat_cols = [tf.feature_column.numeric_column('x',shape=[13])]
deep_model = estimator.DNNClassifier(hidden_units=[13,13,13], feature_columns = feat_cols, n_classes=3, optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01))
input_fn = estimator.inputs.numpy_input_fn(x={'x':scaled_x_train}, y=y_train, shuffle=True, batch_size=10, num_epochs=5)
deep_model.train(input_fn=input_fn, steps=500)
input_fn_eval = estimator.inputs.numpy_input_fn(x={'x':scaled_x_test}, shuffle=False)
preds = list(deep_model.predict(input_fn=input_fn_eval))
predictions = [p['class_ids'][0] for p in preds]

print(classification_report(y_test, predictions))

# Keras
dnn_keras_model = models.Sequential()
dnn_keras_model.add(layers.Dense(units=13, input_dim=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=3, activation='softmax'))

dnn_keras_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

dnn_keras_model.fit(scaled_x_train, y_train, epochs=50)
predictions = dnn_keras_model.predict_classes(scaled_x_test)
print(classification_report(predictions,y_test))

# layers
onehot_y_train = pd.get_dummies(y_train).as_matrix()
onehot_y_test = pd.get_dummies(y_test).as_matrix()
num_feat = 13
num_hidden1 = 13
num_hidden2 = 13
num_outputs = 3
learning_rate= 0.01
X = tf.placeholder(tf.float32, shape=[None, num_feat])
y_true = tf.placeholder(tf.float32, shape=[None, 3])
actf = tf.nn.relu
hidden1 = fully_connected(X, num_hidden1, activation_fn=actf)
hidden2 = fully_connected(hidden1, num_hidden2, activation_fn=actf)
output = fully_connected(hidden2, num_outputs)
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
training_steps = 2
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_steps):
        sess.run(train, feed_dict={X:scaled_x_train,y_true:onehot_y_train})
    logits = output.eval(feed_dict={X:scaled_x_test})
    preds = tf.argmax(logits, axis=1)
    results = preds.eval()
print(classification_report(results, y_test))
