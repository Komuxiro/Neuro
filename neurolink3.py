"""

Решение задачи регрессии с помощью нейроной сети.
Определение стоимости недвижимости из набора данных California Housing Data Set.

**Описание данных**
California Housing содержит данные о средней стоимость домов в Калифорнии для квартала. Файл с данными содержит
следующие столбцы:
*   **longitude** - долгота квартала с недвижимостью.
*   **latitude** - широта квартала с недвижимостью.
*   **housing_median_age** - медиана возраста домов в квартале.
*   **total_rooms** - общее колиичество комнат в квартале.
*   **total_bedrooms** - общее количество спален в квартале.
*   **population** - население квартала.
*   **households** - количество "домохозяйств" в квартале (групп людей живущих вместе в одном доме. Как правило
    это семьи).
*   **median_income** - медианный доход в квартале.
*   **median_house_value** - медианная стоимость дома в квартале.

**Постановка задачи регрессии**
Необходимо определить медианную стоимость дома в квартале, зная все остальные признаки.
**Целевая переменная**: `median_house_value`
**Признаки**: `longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households,
median_income`
Признаки подаются на вход нейронной сети, на выходе сеть должна выдать значение целевой переменной -
`median_house_value`

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузим данные для обучения
train = pd.read_csv('california_housing_train_new.csv')

# Загрузим данные для тестирования
test = pd.read_csv('california_housing_test_new.csv')

# Создадим целевую переменную и признаки
# В моих файлах данных параметров нет - 'longitude', 'latitude',
features = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
            'median_income']
target = 'median_house_value'

# Подготовка данных для обучения.
# Делим набор данных на признаки и правильные ответы.
# Выделяем данные для обучения и преобразуем их в массивы numpy
x_train = train[features].values
x_test = test[features].values

# Выделяем правильные ответы и преобразуем их в массивы numpy
y_train = train[target].values
y_test = test[target].values

# Стандартизация данных
# Вычитаем среднее значение и делим на стандартное отклонение
# Среднее значение
mean = x_train.mean(axis=0)

# Стандартное отклонение
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

# Создаем нейронную сеть
# Выходной слой с одним линейным нейроном - для задачи регрессии функция активации - RELU.
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))

# Компилируем сеть
# Функция ошибки - среднеквадратичное отклонение. Метрика - среднее абсолютное отклонение.
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Обучаем нейронную сеть
history = model.fit(x_train, y_train, epochs=100, validation_split=0.1, verbose=2)

# Визуализация качества обучения
plt.plot(history.history['mae'], label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(history.history['val_mae'], label='Средняя абсолютная ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

# Проверяем работу модели на тестовом наборе данных
scores = model.evaluate(x_test, y_test, verbose=1)
print('Средняя абсолютная ошибка на тестовых данных:', round(scores[1], 4))

# Используем модель для предсказаний
# Выполняем предсказание для тестовой выборки
prediction = model.predict(x_test).flatten()
print(prediction)

# Печатаем примеры результатов
test_index = 0
print('Предсказанная стоимость:', prediction[test_index], ', правильная стоимость:', y_test[test_index])

# Визуализируем результаты предсказаний
# График предсказаний
plt.scatter(y_test, prediction)
plt.xlabel('Правильное значение, $1K')
plt.ylabel('Предсказания, $1K')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
plt.show()

# Гистограмма ошибок
error = prediction-y_test
plt.hist(error, bins=50)
plt.xlabel('Значение ошибка, $1K')
plt.ylabel('Колличество')
plt.show()
