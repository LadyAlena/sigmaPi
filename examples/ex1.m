clear, clc;
close all;

addpath('..\customLayer\');

% Создание слоя с 2 входами, степенью 4 и 10 нейронами
spl = sigmaPiLayer(2, 4, 10, 'sigmaPi1');

% Проверка параметров (должно быть 15 термов)
disp(spl.numTerms); % 15
disp(size(spl.Weights)); % [10, 15]

% Прямой проход
X = rand(2, 100); % 2 признака, 100 образцов
Z = spl.predict(X); % Выход: [10, 100]
disp(size(Z)); % [10, 100]