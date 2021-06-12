import inline
import numpy
import matplotlib.pyplot

# Откроем и прочитаем файл с тестовыми данными
data_file = open('mnist_train_100.csv', 'r')
data_list = data_file.readlines()
data_file.close()

all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)

# количесвто выходных узлов - 10(пример)
onodes = 10
targets = numpy.zeros(onodes) + 0.01  # создадим массив заполненный нулями
targets[int(all_values[0])] = 0.99  # выберем первый элемент и преобразуем его в цело число
