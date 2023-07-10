import tensorflow as tf
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
import warnings


df = pd.read_csv("dataset.csv")

df.head()

plt.scatter(df["X"],df["y"])
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=[1]))
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.01))

model.summary()


history = model.fit(df["X"], df["y"], epochs=300)

plt.plot(history.history['loss'])
plt.show()

df["prediction"] = model.predict(df['X'].values)

plt.scatter(df["X"], df["y"])
plt.plot(df["X"], df["prediction"], color='r')
plt.show()



