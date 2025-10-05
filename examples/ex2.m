clear, clc;
close all;

addpath('..\customLayer\');

%% Генерация данных
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

%% Разделение на обучающую, валидационную и тестовую выборки
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

%% Нормализация данных
% Нормализация входных данных (Z-score нормализация)
data_mean = mean(train_data);
data_std = std(train_data);

train_data_normalized = (train_data - data_mean) ./ data_std;
val_data_normalized = (val_data - data_mean) ./ data_std;
test_data_normalized = (test_data - data_mean) ./ data_std;

% Нормализация целевых переменных (опционально, но рекомендуется для регрессии)
target_mean = mean(train_target);
target_std = std(train_target);

train_target_normalized = (train_target - target_mean) ./ target_std;
val_target_normalized = (val_target - target_mean) ./ target_std;
% test_target не нормализуем, т.к. будем сравнивать с денормализованными предсказаниями

%% Создание архитектуры сети с несколькими ΣΠ-слоями
layers = [
    featureInputLayer(2, 'Name', 'input')   % Входной слой
    
    % Первый ΣΠ-слой
    sigmaPiLayer(2, 2, 15, 'sigmaPiLayer1')

    reluLayer()
    
    % Второй ΣΠ-слой
    sigmaPiLayer(15, 1, 1, 'sigmaPiLayer2')
    
    regressionLayer('Name', 'output')
];

%% Настройка опций обучения с валидацией
options = trainingOptions('adam', ...
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
    'ValidationFrequency', 10);       % Останавливаем, если 5 раз подряд валидационная ошибка растет

%% Обучение сети на нормализованных данных
net = trainNetwork(train_data_normalized, train_target_normalized, layers, options);

%% Оценка качества на тестовой выборке
% Предсказание на нормализованных тестовых данных
test_pred_normalized = predict(net, test_data_normalized);

% Денормализация предсказаний
test_pred = test_pred_normalized * target_std + target_mean;

% Вычисление ошибки
mse = mean((test_pred - test_target).^2);
rmse = sqrt(mse);
mae = mean(abs(test_pred - test_target));
fprintf('Тестовые метрики:\nRMSE = %.4f\nMAE = %.4f\n', rmse, mae);

% Дополнительно: метрики на валидационной выборке
val_pred_normalized = predict(net, val_data_normalized);
val_pred = val_pred_normalized * target_std + target_mean;
val_rmse = sqrt(mean((val_pred - val_target).^2));
fprintf('Валидационные метрики:\nRMSE = %.4f\n', val_rmse);

%% Визуализация результатов
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
xlabel('x'); 
ylabel('y'); 
zlabel('z');

subplot(1,3,2);
surf(x_grid, y_grid, z_pred_grid, 'EdgeColor', 'none');
title('Аппроксимация нейросетью');
xlabel('x'); 
ylabel('y'); 
zlabel('z');

subplot(1,3,3);
error_grid = abs(z_pred_grid - z_true_grid);
surf(x_grid, y_grid, error_grid, 'EdgeColor', 'none');
title('Абсолютная ошибка');
xlabel('x'); 
ylabel('y'); 
zlabel('Ошибка');
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