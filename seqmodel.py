import tensorflow as tf


model = tf.keras.Sequential([tf.keras.layers.Dense(units=16,input_shape=(1,),activation='relu'),
                             tf.keras.layers.Dense(units=32,activation='relu'),
                            tf.keras.layers.Dense(units=2,activation='softmax')
                             ])
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x=scaled_train)