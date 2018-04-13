import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

df = pd.read_csv("cal_housing_clean.csv")
y_val = df["medianHouseValue"]
x_data = df.drop("medianHouseValue", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_val, test_size=0.3, random_state=101)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(data=scaler.transform(X_train),
                       columns=X_train.columns,
                       index=X_train.index)
X_test = pd.DataFrame(data=scaler.transform(X_test),
                      columns=X_test.columns,
                      index=X_test.index)

data_set = []

for i in range(len(df.columns)):
    if df.columns[i] != "medianHouseValue":
        data_set.append(tf.feature_column.numeric_column(df.columns[i]))

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=8, num_epochs=1000, shuffle=True)
model = tf.estimator.DNNRegressor(hidden_units=[6,6,6], feature_columns = data_set)
model.train(input_fn = input_func, steps = 200000)
predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)

predict = []
for i in predictions:
    predict.append(i["predictions"])
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, predict)**0.5)
