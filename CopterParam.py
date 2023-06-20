import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(2,), name='input'),
    # Первый скрытый слой со 10 нейронами
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu', 
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
    tf.keras.layers.Dense(1, name = 'output')
])


model.summary()
# Компилируем модель
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanAbsoluteError()])

X_test = np.array(np.random.random((3, 2)))
Y_test = np.array(np.random.random((3)))
X = [[0, 0], [1, 0], [0, 1], [1, 1]]
Y = [[0], [1], [1], [0]]
print(X, Y)

# Если ошибка не уменьшается на протяжении указанного количества эпох, то процесс обучения прерывается и модель инициализируется весами с самым низким показателем параметра "monitor"
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # указывается параметр, по которому осуществляется ранняя остановка. Обычно это функция потреть на валидационном наборе (val_loss)
    patience=2, # количество эпох по истечении которых закончится обучение, если показатели не улучшатся
    mode='min', # указывает, в какую сторону должна быть улучшена ошибка
    restore_best_weights=True # если параметр установлен в true, то по окончании обучения модель будет инициализирована весами с самым низким показателем параметра "monitor"
)

# Сохраняет модель для дальнейшей загрузки
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='XOR_model', # путь к папке, где будет сохранена модель
    monitor='val_loss',
    save_best_only=True, # если параметр установлен в true, то сохраняется только лучшая модель
    mode='min'
)

# Сохраняет логи выполнения обучения, которые можно будет посмотреть в специальной среде TensorBoard
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='XOR_log', # путь к папке где будут сохранены логи
)

# Обучение
model.fit(X,Y,epochs = 100,
            callbacks = [
            early_stopping,
            model_checkpoint,
            tensorboard]) 

model.save("XOR_model")


predictions = model.predict([[0,0]])
print(predictions)