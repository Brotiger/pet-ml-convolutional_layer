# Модель распознания рукописных цифр
# Программа выводит результат обучения и тестирования
# Датасет для обучения состоит из 1000 элементов
# Датасет для тестирования состоит из 10000 элементов

import numpy as np, sys
np.random.seed(1)

from keras.datasets import mnist

# Загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Формирование датасета для обучения, изменении размера изображений
images, labels = (x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000])

# Преобразование цифр из labels в матрицы вида: [0, 1, 0, 0, 0, 0, 0, 0, 0]
one_hot_labels = np.zeros((len(labels), 10))
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

# Формирование датасета для тестирования, изменении размера изображений
test_images = x_test.reshape(len(x_test), 28*28) / 255

# Преобразование цифр из test_labels в матрицы вида: [0, 1, 0, 0, 0, 0, 0, 0, 0]
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

# Функция активации для скрытого слоя
def tanh(x):
    return np.tanh(x)

# Производная функции активации tanh
def tanh2deriv(output):
    return 1 - (output ** 2)

# Функция активации для выходного слоя
def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

# Гипер параметры
alpha = 2
iterations = 300 
num_labels = 10
batch_size = 128

# Размер изображения
input_rows = 28
input_cols = 28
pixels_per_image = 784

# Размер фильтра (сверточного ядра)
kernel_rows = 3
kernel_cols = 3

# Количество фильтров
num_kernels = 16

# рассчитывает количество позиций, в которых каждое ядро свертки может быть применено к входному изображению без выхода за его границы
hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels

# Веса сверточного слоя (9, 16)
kernels = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01

# Веса выходного слоя (10000, 10)
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

# Извлекает часть изображения
def get_image_section(layer, row_from, row_to, col_from, col_to):
    # глубина z, позиция x, позиция y
    section = layer[:,row_from:row_to,col_from:col_to]

    # -1 - автоматическое определение количества элементов в результирующем массиве,
    # остальные его аргументы это размерность. 
    # 1 - вторым аргументом нужна для дальнейшей конкатенации массивов
    
    section = section.reshape(-1, 1, row_to - row_from, col_to - col_from)
    return section

for j in range(iterations):
    correct_cnt = 0
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
        layer_0 = images[batch_start:batch_end]
        
        # Превращаем входные данные из массива векторов (128, 784) в матрицу (128, 28, 28)
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

        # Формируем массив из 625 матриц представляющих фрагменты изображений (128, 1, 3, 3)
        sects = list()
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start + kernel_cols)
                sects.append(sect)
        
        # Меняем формат с (625, 128, 1, 3, 3) на (128, 625, 3, 3)
        expanded_input = np.concatenate(sects, axis=1)
        
        # Разворачиваем сверточный слой
        # (128, 625, 3, 3) в (80000, 9)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)
        
        # kernel_output - слой между layer_0 и layer_1, kernels - маска/фильтр
        # (80000, 9) * (9, 16) = (80000, 16)
        # Если убрать пачки и прогонять по 1 записи: (625, 9) * (9, 16) = (625, 16)
        kernel_output = flattened_input.dot(kernels)

        # Прогоняем через функцию активации скрытого слоя матрицу преобразованную из (80000, 16) в (128, 10000)
        # 80000 = 625 * 128, 625 - возможные комбинаций 1 маски на 128 - размер пачки изображений
        # 16 - количество масок
        # в
        # 128 - количество масок
        # 10000 - размер скрытого слоя
        layer_1 = tanh(kernel_output.reshape(es[0], -1))

        # Для избежания переобучения случайный образом отключаем половину нейронов,
        # вторую половину усиливаем в 2 раза
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2

        # softmax((128, 10000) * (10000, 10) = (128, 10))
        layer_2 = softmax(np.dot(layer_1, weights_1_2))

        for k in range(batch_size):
            labelset = labels[batch_start + k: batch_start + k + 1]
            correct_cnt += int(np.argmax(layer_2[k:k + 1]) == np.argmax(labelset))
        
        # Чем больше batch_size тем менее точен размер сдвига, по этому делим delta на batch_size для уменьшения сдвига, что бы сеть не ушла в разброс
        # delta - чистая ошибка
        layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask

        # layer_1 * delta = направление и размер сдвига
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)

        l1_d_reshape = layer_1_delta.reshape(kernel_output.shape)
        k_update = flattened_input.T.dot(l1_d_reshape)
        kernels -= alpha * k_update
        
    test_correct_cnt = 0

    for i in range(len(test_images)):
        layer_0 = test_images[i:i + 1]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        layer_0.shape

        sects = list()
        for row_start in range(layer_0.shape[1]-kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start + kernel_rows)
                sects.append(sect)
        
        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        layer_2 = np.dot(layer_1, weights_1_2)

        test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))
    
    if (j % 10 == 0):
        sys.stdout.write("\nI:" + str(j) + " Test-Acc:" + str(test_correct_cnt/float(len(test_images))) + " Train-Acc: " + str(correct_cnt/float(len(images))))