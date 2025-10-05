classdef sigmaPiLayer < nnet.layer.Layer
    properties (Learnable)
        Weights % Матрица весов [num_neurons × num_terms]
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
            %           'random' (по умолчанию), 'xavier', 'he', 'zeros', 'ones'
            
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
            % Генерация всех возможных мультииндексов степени ≤ maxDegree
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
                    % Случайная инициализация из равномерного распределения [-1, 1] (ПО УМОЛЧАНИЮ)
                    weights = 2 * rand(numOut, numIn, 'single') - 1;
                    
                case 'xavier'
                    % Xavier/Glorot инициализация (для линейных активаций)
                    variance = 2 / (numIn + numOut);
                    bound = sqrt(3 * variance);
                    weights = bound * (2 * rand(numOut, numIn, 'single') - 1);
                    
                case 'he'
                    % He инициализация (для ReLU и подобных)
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