# Установка пакета network

1) Скачиваете с репозитория

2) Создаёте venv

3) Внутри venv прописываете pip install "путь до скачанного пакета"

# Мини-обзор

1)Два модуля: core, load.

2)В core написана сама сеть (объект NeuralNetwork) и слои DenseLayer, ConvLayer, MaxPoolingLayer, FlattenLayer

3)В load, чтобы не делать много работы, скачиваются данные с mnist и предобрабатываются (поэтому в setup.py mnist и 
указан в зависимостях, не лучшее решение, но может кому-то будет от этого легче)

# Пояснение

1) Архитектура сети:

 а) Входные данные - можно как список развёрнутых картинок, так и сами картинки

 б) Переменное количество слоёв, теперь никаких параметров, в конструктор передаются слои

 в) Вы можете строить любую архитектуру свёрточной сети, которую вы захотите (конечно, всё ещё стоит следить за корректными входными данными)

2) Инициализация весов в слоях - равномерное распределение( в будущем возможно большее количество различных инициализаций)

3) Также, в будущем возможно большее количество алгоритмов оптимизации нейронных сетей


