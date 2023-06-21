import tensorflow as tf
import numpy as np
import os



testOutVal = []
testVectorMerge = [[]]

print(testVectorMerge)
print(testOutVal)
res = os.listdir("TestVectors/")
print(res)
for files in res:
    testVector = []
    with open('TestVectors/'+ files) as testUnit:
        for line in testUnit:
            arr = line.split()
            strRange = 5
            if(len(arr) == 8):
                strRange = 6
            for idx in range(2, strRange):
                testVector.append(float(arr[idx]))
    testVectorMerge.append(testVector)
    version = files.split('_')
    testOutVal.append(0.003 + int(version[0])*0.0002)

testVectorMerge.remove([])
print(testVectorMerge[0])
print(testOutVal)


model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(50000,), name='input'),
    # Три скрытых слой со 100 нейронами
    tf.keras.layers.Dense(1000, input_shape=(50000,), activation='relu', 
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
    tf.keras.layers.Dense(1000, activation='relu', 
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
    tf.keras.layers.Dense(1000, activation='relu', 
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),                     
    # Выходной слой со 1 нейроном 
    tf.keras.layers.Dense(1),
])


model.summary()
# Компилируем модель
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanAbsoluteError()])


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
    log_dir='CopterTrain_log', # путь к папке где будут сохранены логи
)

# Обучение
model.fit(testVectorMerge,testOutVal,epochs = 300,
            callbacks = [
            early_stopping,
            model_checkpoint,
            tensorboard]) 

model.save("Copter_model")


predictions = model.predict([testVectorMerge[1]])
print(predictions)