# Sigma-Pi ����

Sigma-Pi ���� ($\Sigma\Pi$-����) � ��� ��� ������������� ��������� �����, � ������� ������ ������ ��������� ���������� ����� ������������ ������� ��������.

## �������� ����

� ������� ������� �������� �������� ����������� ��� ���������� ����� ������, ����������� ����� ������� ���������:

$$y = \sigma\left(\sum_i w_i x_i\right)$$

� Sigma-Pi ������� ����������� ������������ ������, ��� ��������� ������������ �������������� ����� �������� ����������:

$$y = \sigma\left(\sum_i w_i \prod_{j \in S_i} x_j\right)$$

����� ���������� ����� ������� �� ��������� ������� ������������ (�������), � ���� $w_i$ ���������� �� ��� ������.

## �������������� �����

����� � ��� ���� ������� ������ $\mathbf{x} = (x_1, x_2, \ldots, x_n)$. ����� ����� Sigma-Pi ������� ����� �������� ���:

$$y = \sigma\left( \sum_{k=1}^{K} w_k \prod_{j \in S_k} x_j^{\alpha_{kj}} \right)$$

���:
- $S_k$ � ��������� ��������, �������� � $k$-� �����,
- $\alpha_{kj}$ � �������, � ������� ���������� $x_j$ � $k$-� ������ (������ ����� ���������������),
- $K$ � ����� ���������� �������.

������� ������ � ����� ������� ����� 0 ��� 1, ����� �� ����� ������ ������������ ����������� ������.

## ��������� �������

��� ��������� ����� ������ $n$ � ������������ ������� $d$ �� ���������� ��� ��������� ������������� $(\alpha_1, \alpha_2, \ldots, \alpha_n)$ �����, ���:

$$\alpha_1 + \alpha_2 + \cdots + \alpha_n \leq d$$

������ ������������ ���������� ����� (�������������� ����):

$$x_1^{\alpha_1} \cdot x_2^{\alpha_2} \cdot \ldots \cdot x_n^{\alpha_n}$$

���������� ����� ������� (��������� ����������) ������ ��� $O(n^d)$, ��� ������������ ������������ ���������� ��� ������� $n$ � $d$.


# ���������� �� MATLAB

��������� ���� `sigmaPiLayer` ��� �������� `nnet.layer.Layer`.

������ ���������� Sigma-Pi ����� ������������� ���������� ��� ������� ����� ������������� � �������������, ��� ����� �������������� ����������� ����� �����������.

```matlab
classdef sigmaPiLayer < nnet.layer.Layer
    % Sigma-Pi ���� ��� ��������� �����
    % ����� ����: Z = Weights * (��������)
    
    properties (Learnable)
        Weights % ������� ����� [num_neurons x num_terms]
    end
    
    properties
        degree        % ������������ ������� ���������
        num_neurons   % ���������� ��������
        numChannels   % ���������� ������� �������
        multiIndices  % ������� ��������������
        numTerms      % ���������� �������������� ������
    end
    
    methods
        function layer = sigmaPiLayer(numChannels, degree, num_neurons, name)
            % ����������� ����
            % �����:
            %   numChannels - ���������� ������ (���������)
            %   degree - ������������ ������� ��������
            %   num_neurons - ���������� �������� � ����
            %   name - ��� ����
            
            layer.Name = name;
            layer.Description = "Sigma-Pi layer (degree " + degree + ") with " + num_neurons + " neurons";
            layer.numChannels = numChannels;
            
            % ��������� ��������������
            layer.multiIndices = sigmaPiLayer.generateMultiIndices(numChannels, degree);
            layer.numTerms = size(layer.multiIndices, 1);
            layer.degree = degree;
            layer.num_neurons = num_neurons;
            
            % ������������� ����� (����� Xavier/Glorot)
            numIn = layer.numTerms;
            numOut = layer.num_neurons;
            variance = 2 / (numIn + numOut);
            bound = sqrt(3 * variance);
            layer.Weights = bound * (2 * rand(numOut, numIn, 'single') - 1);
        end
        
        function Z = predict(layer, X)
            % ������ ������ ����� ����
            % ����:
            %   X - ������� ������ [h, w, c, n] ��� [features, n]
            % �����:
            %   Z - ����� ���� [num_neurons, n]
            
            % �������������� ����� ������ [features, samples]
            if ndims(X) == 4
                [h, w, c, n] = size(X);
                X = reshape(X, [h * w * c, n]);
            elseif ndims(X) == 3
                [h, w, n] = size(X);
                X = reshape(X, [h * w, n]);
            end
            
            % ���������� �������������� ������
            terms = ones(layer.numTerms, size(X, 2), 'like', X);
            for i = 1:layer.numTerms
                for j = 1:layer.numChannels
                    exp_val = layer.multiIndices(i, j);
                    if exp_val > 0
                        terms(i, :) = terms(i, :) .* (X(j, :) .^ exp_val);
                    end
                end
            end
            
            % �������� ����������
            Z = layer.Weights * terms;
        end
    end
    
    methods (Static, Access = private)
        function indices = generateMultiIndices(numVars, maxDegree)
            % ��������� ���� ��������� �������������� ������� ? maxDegree
            % �����:
            %   numVars - ���������� ����������
            %   maxDegree - ������������ �������
            % �����:
            %   indices - ������� [numTerms, numVars], ������ ������ - ������������
            
            grids = cell(1, numVars);
            [grids{:}] = ndgrid(0:maxDegree);
            indices = cell2mat(cellfun(@(g) g(:), grids, 'UniformOutput', false));
            totalDegree = sum(indices, 2);
            indices = indices(totalDegree <= maxDegree, :);
        end
    end
end
```

## �������� ����������

### �������� ����
- `Weights`: ��������� ����, ������� �������� `[num_neurons x num_terms]`.
- `degree`: ������������ ������� ��������.
- `num_neurons`: ���������� �������� � ����.
- `numChannels`: ���������� ������� ���������.
- `multiIndices`: ������� ��������������, ������ ������ ������� ���������� ���� �����.
- `numTerms`: ���������� ������� (�������������� ������).

### ������
**�����������:**
- ������ ��� � �������� ����;
- ���������� ������������� ��� ���� ������� ������� �� ���� `degree`;
- �������������� ���� ������� Xavier/Glorot.

**predict:**
- ����������� ���� � ��������� ������ `[��������, �������]`;
- ��������� ��� ������ (�������������� �����) ��� ������� �������;
- �������� ������� ����� �� ������� ������.

**generateMultiIndices** (�����������, ���������):
- ���������� ��� ���������� �������� �� 0 �� `maxDegree` ��� ������� ��������;
- ��������� ������ ��, � ������� ����� �������� (����� �������) �� ��������� `maxDegree`;

# ������� �������������

## ������ 0: ������� ������������� ����

```matlab
clear, clc;
close all;

% ��������� ���� � ����� � ����������������� ������
addpath('..\customLayer\');

% �������� ���� � 2 �������, �������� 4 � 10 ���������
spl = sigmaPiLayer(2, 4, 10, 'sigmaPi1');

% �������� ���������� (������ ���� 15 �������)
disp(spl.numTerms); % 15
disp(size(spl.Weights)); % [10, 15]

% ������ ������
X = rand(2, 100); % 2 ��������, 100 ��������
Z = spl.predict(X); % �����: [10, 100]
disp(size(Z)); % [10, 100]
```

���� ������ �������������:
- �������� Sigma-Pi ���� � 2 �������� ��������, ������������ �������� �������� - 4 � ����������� �������� - 10;
- �������������� ������ ���������� �������������� ������ (15 ������);
- �������� ������� ������� ����� ����.

## ������ 1: ������������� ������� ���� ����������

���������� ������ ������������� �������:

```matlab
function z_output = func1_z(x, y)
    z_output = abs(x) + abs(y.^2) + abs(x.*y);
end
```

### ��������� ������

```matlab
clear, clc;
close all;

% �������� ��������
x_min = -2;
x_max = 2;
y_min = -2;
y_max = 2;
num_points = 10000;

% ��������� �����
x = (x_max - x_min) * rand(num_points, 1) + x_min;
y = (y_max - y_min) * rand(num_points, 1) + y_min;
z = func1_z(x, y);

% ������������ ��������
data = [x, y];
target = z;
```

### ���������� �� ������� � ������������

```matlab
% ���������� �� ���������, ������������� � �������� �������
rng(42);
indices = randperm(num_points);
train_ratio = 0.7;    % 70% ���������
val_ratio = 0.15;     % 15% �������������
num_train = round(train_ratio * num_points);
num_val = round(val_ratio * num_points);

% ������� ��� ������ �������
train_idx = indices(1:num_train);
val_idx = indices(num_train+1:num_train+num_val);
test_idx = indices(num_train+num_val+1:end);

% ���������� ������
train_data = data(train_idx, :);
train_target = target(train_idx);
val_data = data(val_idx, :);
val_target = target(val_idx);
test_data = data(test_idx, :);
test_target = target(test_idx);

% ������������ ������� ������ (Z-score ������������)
data_mean = mean(train_data);
data_std = std(train_data);

train_data_normalized = (train_data - data_mean) ./ data_std;
val_data_normalized = (val_data - data_mean) ./ data_std;
test_data_normalized = (test_data - data_mean) ./ data_std;

% ������������ ������� ����������
target_mean = mean(train_target);
target_std = std(train_target);

train_target_normalized = (train_target - target_mean) ./ target_std;
val_target_normalized = (val_target - target_mean) ./ target_std;
% test_target �� �����������, �.�. ����� ���������� � ������������������ ��������������
```

### �������� ����������� ����

```matlab
layers = [
    featureInputLayer(2, 'Name', 'input')   % ������� ����
    
    % ������ ??-����
    sigmaPiLayer(2, 2, 15, 'sigmaPiLayer1')

    reluLayer()
    
    % ������ ??-����
    sigmaPiLayer(15, 2, 1, 'sigmaPiLayer2')
    
    regressionLayer('Name', 'output')
];
```

### ��������� ����� �������� � ��������

```matlab
% ��������� ����� �������� � ����������
options = trainingOptions('adam', ...
    'MaxEpochs', 35, ...
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

% �������� ���� �� ��������������� ������
net = trainNetwork(train_data_normalized, train_target_normalized, layers, options);
```

### ������ �������� � ������������

```matlab
% ������ �������� �� �������� �������
test_pred_normalized = predict(net, test_data_normalized);
test_pred = test_pred_normalized * target_std + target_mean;

% ���������� ������
mse = mean((test_pred - test_target).^2);
rmse = sqrt(mse);
mae = mean(abs(test_pred - test_target));
fprintf('�������� �������:\nRMSE = %.4f\nMAE = %.4f\n', rmse, mae);

% ������������ �����������
% �������� ����� ��� ������������
[x_grid, y_grid] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
grid_points = [x_grid(:), y_grid(:)];

% ������������ ����� �����
grid_points_normalized = (grid_points - data_mean) ./ data_std;

% ������������ �� ��������������� �����
z_pred_normalized = predict(net, grid_points_normalized);

% �������������� ������������
z_pred = z_pred_normalized * target_std + target_mean;
z_pred_grid = reshape(z_pred, size(x_grid));

% �������� �������
z_true_grid = func1_z(x_grid, y_grid);

% ���������� ��������
figure;
subplot(1,3,1);
surf(x_grid, y_grid, z_true_grid, 'EdgeColor', 'none');
title('�������� �������');
xlabel('x'); ylabel('y'); zlabel('z');

subplot(1,3,2);
surf(x_grid, y_grid, z_pred_grid, 'EdgeColor', 'none');
title('������������� ����������');
xlabel('x'); ylabel('y'); zlabel('z');

subplot(1,3,3);
error_grid = abs(z_pred_grid - z_true_grid);
surf(x_grid, y_grid, error_grid, 'EdgeColor', 'none');
title('���������� ������');
xlabel('x'); ylabel('y'); zlabel('������');
colorbar;

% ������ ��������� ������������ � ��������� ����������
figure;
plot(test_target, test_pred, 'bo', 'MarkerSize', 4);
hold on;
plot([min(test_target), max(test_target)], [min(test_target), max(test_target)], 'r-', 'LineWidth', 2);
xlabel('�������� ��������');
ylabel('������������� ��������');
title('��������� ������������ �� �������� �������');
legend('������������', '��������� �����', 'Location', 'best');
grid on;
```

# ������ �� �������������

1. **����� ������� ��������**: ��������� � ����� �������� (2-3), ����� �������� ������������ � ��������� �������������� ���������.

2. **�������������**: ����������� L2-������������� � `trainingOptions`:
   ```matlab
   options = trainingOptions('adam', ...
       'L2Regularization', 0.001, ... % ��������� L2-�������������
       ... % ������ ���������
   );
   ```

3. **������������ ������**: ������ ������������ ������� ������, ��� ��� �������������� ����� ����� ������ ���������� ��� ������� ��������� ������.

4. **����������� ����**: ������������ Sigma-Pi ���� � �������� ������������� ������ � ������ ��������� ��� ������� ����� ���������������� � ��������������.

5. **������������������**: Sigma-Pi ���� ����� ���� ����� ���������������, ��� ������� ����, ��� ��� ������ ��� ������������� ������ ����������� ��������������� �����.

6. **�������������� �����������**: ���������� ������� ���� ����� ���������� ��� ���������� ���������� ������ � ������� ��������.