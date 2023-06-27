import tensorflow as tf
import numpy as np

import pandas as pd
import os

TF_ENABLE_ONEDNN_OPTS=0

train_data = tf.data.Dataset.load('sim0/Train/')
test_data = tf.data.Dataset.load('sim0/Test/')
for it in test_data:
    print(it)

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(2600*13,), name='input'),
    # Три скрытых слой со 100 нейронами
    tf.keras.layers.Dense(200, input_shape=(2600*13,), activation='relu'), 
                         # kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
    tf.keras.layers.Dense(200, activation='relu'), 
                          #kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
    tf.keras.layers.Dense(200, activation='relu'), 
                         # kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),                     
    # Выходной слой со 1 нейроном 
    tf.keras.layers.Dense(3),
])


model.summary()
# Компилируем модель
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


# Если ошибка не уменьшается на протяжении указанного количества эпох, то процесс обучения прерывается и модель инициализируется весами с самым низким показателем параметра "monitor"
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # указывается параметр, по которому осуществляется ранняя остановка. Обычно это функция потреть на валидационном наборе (val_loss)
    patience=10, # количество эпох по истечении которых закончится обучение, если показатели не улучшатся
    mode='min', # указывает, в какую сторону должна быть улучшена ошибка
    restore_best_weights=True # если параметр установлен в true, то по окончании обучения модель будет инициализирована весами с самым низким показателем параметра "monitor"
)

# Сохраняет модель для дальнейшей загрузки
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='Copter_model', # путь к папке, где будет сохранена модель
    monitor='val_loss',
    save_best_only=True, # если параметр установлен в true, то сохраняется только лучшая модель
    mode='min'
)

# Сохраняет логи выполнения обучения, которые можно будет посмотреть в специальной среде TensorBoard
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='CopterTrain_log', # путь к папке где будут сохранены логи
) 

# Обучение
model.fit(train_data,validation_data=test_data,epochs = 70,
          callbacks = [
            early_stopping,
            model_checkpoint,
            tensorboard])
model.save("Copter_model")

predictions = model.predict(test_data, verbose=2)
print(predictions)