# Если нет изображения, выполнить sudo pacman -S tk

# библиотека glop поможет выбрать несколько файлов с помощью шаблонов
import glob
# библиотека imageio загрузки данных из изображений
import imageio
# библиотека scipy для сигмовидной функции expit
import scipy.special
import numpy
import matplotlib.pyplot as plt


# Определим класс нейронной сети
class neuralNetwork:
    # инициализируем нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodse, learningrate):
        # Зададим кол-во узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodse

        '''
                создадим матрицы связей между узлами wih (между входным и скрытым слоем) и who (между скрытым и выходным слоем)
                Весовые коэффициенты связей между узлом i и узлом j обозначены как w_i_j:
                w11 w21 w12 w22 и т.д. numpy.random.rand(3, 3) - 0.5
        '''
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, - 0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, - 0.5), (self.onodes, self.hnodes))

        # коэффицент обучения
        self.lr = learningrate

        # используем сигмоиду в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # тренировка нейронной сети
    def train(self, inputs_list, target_list):
        # преобразуем список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # рассчитаем входящие сигналы для скрытого слоя
        hiddden_inputs = numpy.dot(self.wih, inputs)
        # рассчитаем исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hiddden_inputs)

        # рассчитаем входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитаем исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        '''
        Ошибки выходного слоя = (целевое значени- фактическое значени)
        '''
        output_errors = targets - final_outputs
        '''
        Ошибки скрытого слоя - это ошибки output_errors, распределенные пропорционально
        весовым коэффициентам связей и рекомбинированные на скрытых узлах
        '''
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # обновим весовые коэффициенты для связей между скрытыми и выходными слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # обновим весовые коэффициенты для связей между входными и скрытыми слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        #pass

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразуем список входных значений в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # расчитаем входящие сигналы дял скрытог ослоя
        hidden_inputs = numpy.dot(self.wih, inputs)

        # расчитаем исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # расчитаем входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # расчитаем исхожящие сигналы для выходног ослоя
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# колличество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# коэффициент обучения равен 0,1
learning_rate = 0.1

# создать экземпляр нейронной сети
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# загрузим в список тренировочный набор данных
training_data_file = open('mnist_train_100.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# тренировка нейронной сети
# переменная epochs указывает сколько раз тренировочный набор данных используеся для тренировки сети
epochs = 10
for e in range(epochs):
    # перебрать все записи тренировчного набора данных
    for record in training_data_list:
        # получим список значений, используя символы запятой (,) в качестве разделителей
        all_values = record.split(',')

        # масштабируем и сместим выходные значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # создадим целевые выходные значения (все = 0,01, кроме желаемого = 0,99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] - целевое значение для данной записи
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# загрузим в список тестовый набор данных
test_data_file = open('mnist_test_10.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Создадим журнал оценок нейронки, изначально он пуст
scorecard = []

# тестирование нейронной сети
# перебрать все записи тестового набора данных
for record in test_data_list:
    # получим список значений, используя символы запятой (,) в качестве разделителей
    all_values = record.split(',')

    # Правильный ответ - первое значение
    correct_label = int(all_values[0])

    # масштабируем и сместим выходные значения
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # опрос сети
    outputs = n.query(inputs)

    # индекс наибольшего значения является маркерным значением
    label = numpy.argmax(outputs)

    # присоеденить оценку ответа сети к концу списка
    if (label == correct_label):
        # в случае правильного ответа сети, присоеденить к списку значение - 1
        scorecard.append(1)
    else:
        # в случае неправильного ответа сети, присоеденить к списку значение - 0
        scorecard.append(0)
        pass
    pass

# рассчитаем показатель эффективности в виде доли правильных ответов
scorecard_array = numpy.asarray(scorecard)
print("Эффективность = ", scorecard_array.sum() / scorecard_array.size)

all_values = training_data_list[0].split(',')
print(all_values[0])
print(scorecard)

