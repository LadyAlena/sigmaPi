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
    end
    
    methods
        function layer = sigmaPiLayer(numChannels, degree, num_neurons, name)
            layer.Name = name;
            layer.Description = "Sigma-Pi layer (degree " + degree + ") with " + num_neurons + " neurons";
            layer.numChannels = numChannels;
            
            % Генерация мультииндексов
            layer.multiIndices = sigmaPiLayer.generateMultiIndices(numChannels, degree);
            layer.numTerms = size(layer.multiIndices, 1);
            layer.degree = degree;
            layer.num_neurons = num_neurons;
            
            % Инициализация весов Glorot/Xaiver
            numIn = layer.numTerms;
            numOut = layer.num_neurons;
            variance = 2 / (numIn + numOut);
            bound = sqrt(3 * variance);
            layer.Weights = bound * (2 * rand(numOut, numIn, 'single') - 1);
        end
        
        function Z = predict(layer, X)
            % Преобразование входа в 2D формат [features, samples]
            if ndims(X) == 4
                [h, w, c, n] = size(X);
                X = reshape(X, [h * w * c, n]);
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
            
            % Линейная комбинация и активация
            Z = layer.Weights * terms;
        end
    end
    
    methods (Static, Access = private)
        function indices = generateMultiIndices(numVars, maxDegree)
            grids = cell(1, numVars);
            [grids{:}] = ndgrid(0:maxDegree);
            indices = cell2mat(cellfun(@(g) g(:), grids, 'UniformOutput', false));
            totalDegree = sum(indices, 2);
            indices = indices(totalDegree <= maxDegree, :);
        end
    end
end