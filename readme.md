# Sigma-Pi сети

Sigma-Pi сети ($\Sigma\Pi$-сети) — это тип искусственных нейронных сетей, в которых каждый нейрон вычисляет взвешенную сумму произведений входных сигналов.

## Основная идея

В обычном нейроне выходное значение вычисляется как взвешенная сумма входов, пропущенная через функцию активации:

$$y = \sigma\left(\sum_i w_i x_i\right)$$

В Sigma-Pi нейроне добавляются произведения входов, что позволяет моделировать взаимодействия между входными признаками:

$$y = \sigma\left(\sum_i w_i \prod_{j \in S_i} x_j\right)$$

Здесь внутренняя сумма берется по различным наборам произведений (мономам), а веса $w_i$ умножаются на эти мономы.

## Математическая форма

Пусть у нас есть входной вектор $\mathbf{x} = (x_1, x_2, \ldots, x_n)$. Тогда выход Sigma-Pi нейрона можно записать как:

$$y = \sigma\left( \sum_{k=1}^{K} w_k \prod_{j \in S_k} x_j^{\alpha_{kj}} \right)$$

где:
- $S_k$ — множество индексов, входящих в $k$-й моном,
- $\alpha_{kj}$ — степень, в которую возводится $x_j$ в $k$-м мономе (обычно целое неотрицательное),
- $K$ — общее количество мономов.

Частный случай — когда степени равны 0 или 1, тогда мы имеем просто произведения подмножеств входов.

## Генерация мономов

Для заданного числа входов $n$ и максимальной степени $d$ мы генерируем все возможные мультииндексы $(\alpha_1, \alpha_2, \ldots, \alpha_n)$ такие, что:

$$\alpha_1 + \alpha_2 + \cdots + \alpha_n \leq d$$

Каждый мультииндекс определяет моном (полиномиальный терм):

$$x_1^{\alpha_1} \cdot x_2^{\alpha_2} \cdot \ldots \cdot x_n^{\alpha_n}$$

Количество таких мономов (свободных параметров) растет как $O(n^d)$, что ограничивает практическое применение при больших $n$ и $d$.


# Реализация на MATLAB

Реализуем слой `sigmaPiLayer` как подкласс `nnet.layer.Layer`.

Данная реализация Sigma-Pi сетей предоставляет инструмент для решения задач аппроксимации и классификации, где важны полиномиальные зависимости между переменными.

```matlab
classdef sigmaPiLayer < nnet.layer.Layer
    % Sigma-Pi слой для нейронных сетей
    % Выход слоя: Z = Weights x (полиномиальные термы)
    
    properties (Learnable)
        Weights % Матрица весов [num_neurons ? num_terms]
    end
    
    properties
        degree        % Максимальная степень полиномов
        num_neurons   % Количество нейронов
        numChannels   % Количество входных каналов
        multiIndices  % Матрица мультииндексов
        numTerms      % Количество полиномиальных членов
        initialization_method % Метод инициализации весов
    end
    
    methods
        function layer = sigmaPiLayer(numChannels, degree, num_neurons, name, varargin)
            % Конструктор слоя
            % Входы:
            %   numChannels - количество входных каналов (признаков)
            %   degree - максимальная степень полинома
            %   num_neurons - количество нейронов в слое
            %   name - имя слоя
            %   varargin - опциональные параметры:
            %       'Initialization' - метод инициализации:
            %           'random' (по умолчанию), 'xavier', 'he', 'zeros', 'ones', 'small_random'
            
            layer.Name = name;
            layer.Description = "Sigma-Pi layer (degree " + degree + ") with " + num_neurons + " neurons";
            layer.numChannels = numChannels;
            
            % Парсинг дополнительных параметров
            p = inputParser;
            addParameter(p, 'Initialization', 'random', @ischar);
            parse(p, varargin{:});
            
            layer.initialization_method = p.Results.Initialization;
            
            % Генерация мультииндексов
            layer.multiIndices = sigmaPiLayer.generateMultiIndices(numChannels, degree);
            layer.numTerms = size(layer.multiIndices, 1);
            layer.degree = degree;
            layer.num_neurons = num_neurons;
            
            % Инициализация весов выбранным методом
            layer.Weights = sigmaPiLayer.initializeWeights(...
                layer.num_neurons, layer.numTerms, layer.initialization_method);
        end
        
        function Z = predict(layer, X)
            % Прямой проход через слой
            % Вход:
            %   X - входные данные [h, w, c, n] или [features, n]
            % Выход:
            %   Z - выход слоя [num_neurons, n]
            
            % Преобразование входа в 2D формат [features, samples]
            if ndims(X) == 4
                [h, w, c, n] = size(X);
                X = reshape(X, [h * w * c, n]);
            elseif ndims(X) == 3
                [h, w, n] = size(X);
                X = reshape(X, [h * w, n]);
            end
            
            % Вычисление полиномиальных термов
            terms = ones(layer.numTerms, size(X, 2), 'like', X);
            for i = 1:layer.numTerms
                for j = 1:layer.numChannels
                    exp_val = layer.multiIndices(i, j);
                    if exp_val > 0
                        terms(i, :) = terms(i, :) .* (X(j, :) .^ exp_val);
                    end
                end
            end
            
            % Линейная комбинация
            Z = layer.Weights * terms;
        end
    end
    
    methods (Static, Access = private)
        function indices = generateMultiIndices(numVars, maxDegree)
            % Генерация всех возможных мультииндексов степени ? maxDegree
            % Входы:
            %   numVars - количество переменных
            %   maxDegree - максимальная степень
            % Выход:
            %   indices - матрица [numTerms, numVars], каждая строка - мультииндекс
            
            grids = cell(1, numVars);
            [grids{:}] = ndgrid(0:maxDegree);
            indices = cell2mat(cellfun(@(g) g(:), grids, 'UniformOutput', false));
            totalDegree = sum(indices, 2);
            indices = indices(totalDegree <= maxDegree, :);
        end
        
        function weights = initializeWeights(numOut, numIn, method)
            % Инициализация весов выбранным методом
            % Входы:
            %   numOut - количество нейронов (выходной размер)
            %   numIn - количество термов (входной размер)
            %   method - метод инициализации
            
            switch lower(method)
                case 'random'
                    % Случайная инициализация из равномерного распределения [-1, 1] (по умолчанию)
                    weights = 2 * rand(numOut, numIn, 'single') - 1;
                    
                case 'xavier'
                    % Xavier/Glorot инициализация (для линейных и симметричных активаций)
                    variance = 2 / (numIn + numOut);
                    bound = sqrt(3 * variance);
                    weights = bound * (2 * rand(numOut, numIn, 'single') - 1);
                    
                case 'he'
                    % He инициализация (для ReLU и подобных активаций)
                    stddev = sqrt(2 / numIn);
                    weights = stddev * randn(numOut, numIn, 'single');
                    
                case 'small_random'
                    % Маленькие случайные значения
                    weights = 0.01 * randn(numOut, numIn, 'single');
                    
                case 'zeros'
                    % Нулевая инициализация
                    weights = zeros(numOut, numIn, 'single');
                    
                case 'ones'
                    % Инициализация единицами
                    weights = ones(numOut, numIn, 'single');
                    
                otherwise
                    % По умолчанию используем random
                    weights = 2 * rand(numOut, numIn, 'single') - 1;
            end
        end
    end
    
    methods (Static)
        function availableMethods = getInitializationMethods()
            % Возвращает список доступных методов инициализации
            availableMethods = {'random', 'xavier', 'he', 'small_random', 'zeros', 'ones'};
        end
    end
end
```

## Описание реализации

### Свойства слоя
- `Weights`: обучаемые веса, матрица размером `[num_neurons x num_terms]`.
- `degree`: максимальная степень полинома.
- `num_neurons`: количество нейронов в слое.
- `numChannels`: количество входных признаков.
- `multiIndices`: матрица мультииндексов, каждая строка которой определяет один моном.
- `numTerms`: количество мономов (полиномиальных термов).
- `initialization_method`: метод инициализации весов.

### Методы
**Конструктор:**
- задает имя и описание слоя;
- генерирует мультииндексы для всех мономов степени не выше `degree`;
- поддерживает выбор метода инициализации весов через опциональный параметр `'Initialization'`. По умолчанию используется случайная инициализация (`'random'`).

**predict:**
- преобразует вход в двумерный массив `[признаки, истинные метки]`;
- вычисляет все мономы (полиномиальные термы) для каждого примера;
- умножает матрицу весов на матрицу термов.

**generateMultiIndices** (статический, приватный):
- генерирует все комбинации степеней от 0 до `maxDegree` для каждого признака;
- оставляет только те, у которых сумма степеней (общая степень) не превышает `maxDegree`.

**initializeWeights** (статический, приватный):
- реализует различные методы инициализации весов;
- поддерживает методы: `random`, `xavier`, `he`, `small_random`, `zeros`, `ones`.

**getInitializationMethods** (статический):
- возвращает список доступных методов инициализации.

# Примеры использования

## Пример 1: Базовое использование слоя

```matlab
clear, clc;
close all;

% Добавляем путь к папке с пользовательскими слоями
addpath('..\customLayer\');

% Создание слоя с 2 входами, степенью 4 и 10 нейронами
spl = sigmaPiLayer(2, 4, 10, 'sigmaPi1');

% Проверка параметров (должно быть 15 мономов)
disp(spl.numTerms); % 15
disp(size(spl.Weights)); % [10, 15]

% Прямой проход
X = rand(2, 100); % 2 признака, 100 образцов
Z = spl.predict(X); % Выход: [10, 100]
disp(size(Z)); % [10, 100]
```

Этот пример демонстрирует:
- создание Sigma-Pi слоя с 2 входными каналами, максимальной степенью полинома - 4 и количеством нейронов - 10;
- автоматический расчет количества полиномиальных членов (15 термов);
- проверку прямого прохода через слой.

## Пример 2: Аппроксимация функции двух переменных

Рассмотрим задачу аппроксимации функции:

```matlab
function z_output = func1_z(x, y)
    z_output = abs(x) + abs(y.^2) + abs(x.*y);
end
```

### Генерация данных

```matlab
clear, clc;
close all;

% Диапазон значений
x_min = -2;
x_max = 2;
y_min = -2;
y_max = 2;
num_points = 10000;

% Случайные точки
x = (x_max - x_min) * rand(num_points, 1) + x_min;
y = (y_max - y_min) * rand(num_points, 1) + y_min;
z = func1_z(x, y);

% Формирование датасета
data = [x, y];
target = z;
```

### Разделение на выборки и нормализация

```matlab
% Разделение на обучающую, валидационную и тестовую выборки
rng(42);
indices = randperm(num_points);
train_ratio = 0.7;    % 70% обучающих
val_ratio = 0.15;     % 15% валидационных
num_train = round(train_ratio * num_points);
num_val = round(val_ratio * num_points);

% Индексы для разных выборок
train_idx = indices(1:num_train);
val_idx = indices(num_train+1:num_train+num_val);
test_idx = indices(num_train+num_val+1:end);

% Разделение данных
train_data = data(train_idx, :);
train_target = target(train_idx);
val_data = data(val_idx, :);
val_target = target(val_idx);
test_data = data(test_idx, :);
test_target = target(test_idx);

% Нормализация входных данных (Z-score нормализация)
data_mean = mean(train_data);
data_std = std(train_data);

train_data_normalized = (train_data - data_mean) ./ data_std;
val_data_normalized = (val_data - data_mean) ./ data_std;
test_data_normalized = (test_data - data_mean) ./ data_std;

% Нормализация целевых переменных
target_mean = mean(train_target);
target_std = std(train_target);

train_target_normalized = (train_target - target_mean) ./ target_std;
val_target_normalized = (val_target - target_mean) ./ target_std;
% test_target не нормализуем, т.к. будем сравнивать с денормализованными предсказаниями
```

### Создание архитектуры сети

```matlab
layers = [
    featureInputLayer(2, 'Name', 'input')   % Входной слой
    
    % Первый sigmaPi-слой
    sigmaPiLayer(2, 2, 15, 'sigmaPiLayer1', 'Initialization', 'he')

    reluLayer()
    
    % Второй sigmaPi-слой
    sigmaPiLayer(15, 2, 1, 'sigmaPiLayer2', 'Initialization', 'he')
    
    regressionLayer('Name', 'output')
];
```

### Настройка опций обучения и обучение

```matlab
% Настройка опций обучения с валидацией
options = trainingOptions('adam', ...
    'L2Regularization', 0.001, ...
    'MaxEpochs', 25, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 200, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'cpu', ...
    'ValidationData', {val_data_normalized, val_target_normalized}, ...
    'ValidationFrequency', 10);

% Обучение сети на нормализованных данных
net = trainNetwork(train_data_normalized, train_target_normalized, layers, options);
```

### Оценка качества и визуализация

```matlab
% Оценка качества на тестовой выборке
test_pred_normalized = predict(net, test_data_normalized);
test_pred = test_pred_normalized * target_std + target_mean;

% Вычисление ошибки
mse = mean((test_pred - test_target).^2);
rmse = sqrt(mse);
mae = mean(abs(test_pred - test_target));
fprintf('Тестовые метрики:\nRMSE = %.4f\nMAE = %.4f\n', rmse, mae);

% Визуализация результатов
% Создание сетки для предсказания
[x_grid, y_grid] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
grid_points = [x_grid(:), y_grid(:)];

% Нормализация точек сетки
grid_points_normalized = (grid_points - data_mean) ./ data_std;

% Предсказание на нормализованной сетке
z_pred_normalized = predict(net, grid_points_normalized);

% Денормализация предсказаний
z_pred = z_pred_normalized * target_std + target_mean;
z_pred_grid = reshape(z_pred, size(x_grid));

% Истинная функция
z_true_grid = func1_z(x_grid, y_grid);

% Построение графиков
figure;
subplot(1,3,1);
surf(x_grid, y_grid, z_true_grid, 'EdgeColor', 'none');
title('Исходная функция');
xlabel('x'); ylabel('y'); zlabel('z');

subplot(1,3,2);
surf(x_grid, y_grid, z_pred_grid, 'EdgeColor', 'none');
title('Аппроксимация нейросетью');
xlabel('x'); ylabel('y'); zlabel('z');

subplot(1,3,3);
error_grid = abs(z_pred_grid - z_true_grid);
surf(x_grid, y_grid, error_grid, 'EdgeColor', 'none');
title('Абсолютная ошибка');
xlabel('x'); ylabel('y'); zlabel('Ошибка');
colorbar;

% График сравнения предсказаний с истинными значениями
figure;
plot(test_target, test_pred, 'bo', 'MarkerSize', 4);
hold on;
plot([min(test_target), max(test_target)], [min(test_target), max(test_target)], 'r-', 'LineWidth', 2);
xlabel('Истинные значения');
ylabel('Предсказанные значения');
title('Сравнение предсказаний на тестовой выборке');
legend('Предсказания', 'Идеальная линия', 'Location', 'best');
grid on;
```

# Советы по использованию

1. **Выбор степени полинома**: Начинайте с малых степеней (2-3), чтобы избежать переобучения и уменьшить вычислительную сложность.

2. **Выбор метода инициализации**:
   - `random` (по умолчанию) - универсальный метод для большинства случаев;
   - `xavier` (метод Glorot/Xavier) - оптимален для симметричных функций активации (tanh, sigmoid);
   - `he` (метод Kaiming He) - рекомендуется для сетей с функцией ReLU её  вариациями;
   - `small_random`;
   - `zeros`, `ones` - в основном для тестирования и отладки.

3. **Регуляризация**: Используйте L2-регуляризацию в `trainingOptions`:
   ```matlab
   options = trainingOptions('adam', ...
       'L2Regularization', 0.001, ... % Добавляем L2-регуляризацию
       ... % другие параметры
   );
   ```

4. **Нормализация данных**: Всегда нормализуйте входные данные, так как полиномиальные члены могут сильно возрастать при больших значениях входов.

5. **Архитектура сети**: Комбинируйте Sigma-Pi слои с обычными полносвязными слоями для лучшей эффективностью.

6. **Вычислительные ограничения**: Учитывайте быстрый рост числа параметров при увеличении количества входов и степени полинома.