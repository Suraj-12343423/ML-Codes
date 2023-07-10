import numpy as np
import tensorflow as tf
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_labels = []
train_samples = []

#Example data
#A drug was tested on individuals from age 13 to 100
#Trial had 2100 patients,Half over 65 yrs and half under 65 yrs
#95% of patients over 65 no experienced side effects
#95% of patients under 65 no experienced no side effects

for i in range(50):
    #5% of younger indiv wo experienced side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    #5% of older indiv who experienced sideeffects
    random_older = randint(64,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # 95% of younger indiv wo experienced side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # 95% of older indiv who experienced sideeffects
    random_older = randint(64,100)
    train_samples.append(random_older)
    train_labels.append(1)
#for i in train_samples:
    #print(i)

#for i in train_labels:
    #print(i)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels,train_samples = shuffle(train_labels,train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

for i in scaled_train_samples:
    print(i)


model = tf.keras.Sequential([tf.keras.layers.Dense(units=16,input_shape=(1,),activation='relu'),
                             tf.keras.layers.Dense(units=32,activation='relu'),
                            tf.keras.layers.Dense(units=2,activation='softmax')
                             ])
model.summary()
  
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x=scaled_train_samples,y=train_labels,validation_split=0.1,batch_size=10,epochs=30,shuffle=True,verbose=2)

test_labels = []
test_samples = []

for i in range(50):
    #5% of younger indiv wo experienced side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    #5% of older indiv who experienced sideeffects
    random_older = randint(64,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(1000):
    # 95% of younger indiv wo experienced side effects
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # 95% of older indiv who experienced sideeffects
    random_older = randint(64,100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels,test_samples = shuffle(test_labels,test_samples)

scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

predictions = model.predict(x=scaled_test_samples,batch_size=10,verbose=0)
for i in predictions:
    print(i)

rounded_predictions = np.argmax(predictions,axis=1)

#Confusion matrix can be used to visualize how well the model predicts

cm = confusion_matrix(y_true=test_labels,y_pred=rounded_predictions)

