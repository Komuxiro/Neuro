"""
Распознавание объектов на изображениях из набора данных CIFAR-10
"""

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# from google.colab import files

# Названия классов из набора данных CIFAR-10
classes = ['Самолет', ['Автомобиль'], ['Птица'], ['Кот'], ['Олень'], ['Собака'], ['Лягушка'], ['Лошадь'],
           ['Корабль'], ['Грузовик']]

# Загрузим данные
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Просмотр примера данных
plt.figure(figsize=(10, 10))
for i in range(100, 150):
    plt.subplot(5, 10, i - 100 + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(classes[y_train[i][0]])

# Нормализуем данные
x_train = x_train / 255
x_test = x_test / 255

# Преобразуем правильные ответы в формат one hot encoding
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Создаем нейронную сеть
# Создаем последовательную модель
model = Sequential()

# Первый сверточный слой
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
# Второй сверточный слой
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))

# Третий сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu'))
# Четвертый сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))

# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(512, activation='relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.5))
# Выходной полносвязный слой
model.add(Dense(10, activation='softmax'))

# Печатаем информацию о сети (если не нужно это можно удалить)
model.summary()

# Комплируем модель
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучим нейронную сеть
history = model.fit(x_train, y_train, batch_size=128, epochs=25, validation_split=0.1, verbose=2)

# Оценим качество обучения
# Оценим качество на тестовых данных
scores = model.evaluate(x_test, y_test, verbose=1)
print("Доля правильных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))
plt.plot(history.history['accuracy'], label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.show()

# Сохраним обученную нейронную сеть на локальном диске
model.save('cifar10_model.h5')
# files.download('cifar10_model.h5')


# Применим сеть для распознования объектов на изображениях
index = 4065
plt.imshow(x_test[index])
plt.show()

# Преобразуем текстовое изображение
x = x_test[index]
x = np.expand_dims(x, axis=0)

# Запустим распознование и напечатаем результат
prediction = model.predict(x)
print(prediction)

# Преобразуем результат из формата one hot encoding
prediction = np.argmax(prediction)
print(classes[prediction])

# Напечатаем правильный ответ
print(classes[y_test[index][0]])

'''
# Запустим распознование собственных изображений, помметстить изображение рядом с кодом нейронки
img_path = 'plane.jpg'
img = image.load_img(img_path, target_size=(32, 32))
plt.imshow(img)
plt.show()

# Преобразуем картинку в массив данных
x = image.img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)

# Запустим распознование
prediction = model.predict(x)
prediction = np.argmax(prediction)
print(classes[prediction])
'''
