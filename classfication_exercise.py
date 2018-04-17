import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

def fix_income(value):
    if value == ' <=50K':
        return 0
    elif value == ' >50K':
        return 1
df = pd.read_csv("census_data.csv")
df["income_bracket"] = df["income_bracket"].apply(fix_income)

y_val = df["income_bracket"]
x_data = df.drop("income_bracket", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_val, test_size=0.3, random_state=101)

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=8, num_epochs=1000, shuffle=True)

fea_columns = [gender, occupation, marital_status, relationship, education, workclass, native_country, age, education_num, capital_gain, capital_loss, hours_per_week]

model = tf.estimator.LinearClassifier(feature_columns=fea_columns)
model.train(input_fn=input_func, steps = 20000)
predict_input_func=tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)
pred_gen=model.predict(input_fn=predict_input_func)
predictions=list(pred_gen)

predict=[]
for i in predictions:
    predict.append(i["class_ids"][0])
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))
