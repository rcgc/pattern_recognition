% orthogonal_moment_analysis.m
% Análisis de momentos ortogonales usando 20 imágenes por dígito (MATLAB)

ord = 80;
digit = 8;  % Cambiar este valor para otro dígito (0-9)
folder = num2str(digit);
num_imgs = 20;

% Inicialización
legendre_data = [];
chebyshev1_data = [];
chebyshev2_data = [];

% ===== Paso 1: Cargar imágenes y calcular momentos =====
for i = 1:num_imgs
    filename = fullfile(folder, sprintf('%d.png', i));
    F = imread(filename);
    if size(F, 3) == 3
        F = rgb2gray(F);
    end

    % Asegúrate de tener estas funciones definidas
    v_legendre = legmoms_vec(F, ord);
    v_chebyshev1 = dchebmoms_vec(F, ord);
    v_chebyshev2 = cheb2moms_vec(F, ord);

    % Asegurar que se agregan como filas (1xN)
    legendre_data = [legendre_data; v_legendre(:)'];
    chebyshev1_data = [chebyshev1_data; v_chebyshev1(:)'];
    chebyshev2_data = [chebyshev2_data; v_chebyshev2(:)'];
end

% Guardar CSVs
writematrix(legendre_data, ['Legendre_' folder '.csv']);
writematrix(chebyshev1_data, ['Chebyshev_' folder '.csv']);
writematrix(chebyshev2_data, ['Chebyshev2_' folder '.csv']);

% ===== Paso 2: Preparar datos (función local) =====
[legendre_train, legendre_target] = prepare_data(['Legendre_' folder '.csv']);
[chebyshev1_train, chebyshev1_target] = prepare_data(['Chebyshev_' folder '.csv']);
[chebyshev2_train, chebyshev2_target] = prepare_data(['Chebyshev2_' folder '.csv']);

% ===== Paso 3: Random Forest =====
legendre_rf = rf_predict(legendre_train);
chebyshev1_rf = rf_predict(chebyshev1_train);
chebyshev2_rf = rf_predict(chebyshev2_train);

% ===== Paso 4: Red Neuronal =====
legendre_ann = ann_predict(legendre_train, legendre_target);
chebyshev1_ann = ann_predict(chebyshev1_train, chebyshev1_target);
chebyshev2_ann = ann_predict(chebyshev2_train, chebyshev2_target);

% ===== Paso 5: Comparación =====
compare_vectors('Legendre', legendre_target, legendre_rf, legendre_ann);
compare_vectors('Chebyshev1', chebyshev1_target, chebyshev1_rf, chebyshev1_ann);
compare_vectors('Chebyshev2', chebyshev2_target, chebyshev2_rf, chebyshev2_ann);

% ===== Paso 6: Visualización con normalización consistente =====
% Calcular min y max para cada conjunto de vectores para escalarlos igual

% Legendre normalization limits
min_leg = min([legendre_target, legendre_rf, legendre_ann], [], 'all');
max_leg = max([legendre_target, legendre_rf, legendre_ann], [], 'all');

show_image_from_vector(legendre_target, 'Legendre Original', min_leg, max_leg);
show_image_from_vector(legendre_rf, 'Legendre RF Prediction', min_leg, max_leg);
show_image_from_vector(legendre_ann, 'Legendre ANN Prediction', min_leg, max_leg);

% Chebyshev1 normalization limits
min_cheb1 = min([chebyshev1_target, chebyshev1_rf, chebyshev1_ann], [], 'all');
max_cheb1 = max([chebyshev1_target, chebyshev1_rf, chebyshev1_ann], [], 'all');

show_image_from_vector(chebyshev1_target, 'Chebyshev1 Original', min_cheb1, max_cheb1);
show_image_from_vector(chebyshev1_rf, 'Chebyshev1 RF Prediction', min_cheb1, max_cheb1);
show_image_from_vector(chebyshev1_ann, 'Chebyshev1 ANN Prediction', min_cheb1, max_cheb1);

% Chebyshev2 normalization limits
min_cheb2 = min([chebyshev2_target, chebyshev2_rf, chebyshev2_ann], [], 'all');
max_cheb2 = max([chebyshev2_target, chebyshev2_rf, chebyshev2_ann], [], 'all');

show_image_from_vector(chebyshev2_target, 'Chebyshev2 Original', min_cheb2, max_cheb2);
show_image_from_vector(chebyshev2_rf, 'Chebyshev2 RF Prediction', min_cheb2, max_cheb2);
show_image_from_vector(chebyshev2_ann, 'Chebyshev2 ANN Prediction', min_cheb2, max_cheb2);

function [train_data, target_row] = prepare_data(filename)
    data = readmatrix(filename);
    target_row = data(15, :);
    data(15, :) = [];
    train_data = data;
end

function prediction = rf_predict(train_data)
    num_features = size(train_data, 2);
    prediction = zeros(1, num_features);
    for f = 1:num_features
        y = train_data(:, f);
        X = (1:size(train_data, 1))';
        model = TreeBagger(50, X, y, 'Method', 'regression');
        prediction(f) = predict(model, 15);
    end
end

function prediction = ann_predict(train_data, target_row)
    [n_samples, n_features] = size(train_data);
    X = train_data;
    Y = train_data;

    mu = mean(X);
    sigma = std(X);
    X_norm = (X - mu) ./ sigma;
    Y_norm = (Y - mu) ./ sigma;

    hidden_size = 10;
    W1 = randn(hidden_size, n_features) * 0.1;
    b1 = zeros(hidden_size, 1);
    W2 = randn(n_features, hidden_size) * 0.1;
    b2 = zeros(n_features, 1);

    lr = 0.01;
    epochs = 500;

    for epoch = 1:epochs
        for i = 1:n_samples
            x = X_norm(i, :)';
            y = Y_norm(i, :)';

            z1 = W1 * x + b1;
            a1 = 1 ./ (1 + exp(-z1));
            z2 = W2 * a1 + b2;
            y_pred = z2;

            dz2 = y_pred - y;
            dW2 = dz2 * a1';
            db2 = dz2;
            da1 = W2' * dz2;
            dz1 = da1 .* a1 .* (1 - a1);
            dW1 = dz1 * x';
            db1 = dz1;

            W2 = W2 - lr * dW2;
            b2 = b2 - lr * db2;
            W1 = W1 - lr * dW1;
            b1 = b1 - lr * db1;
        end
    end

    x = (target_row - mu) ./ sigma;
    z1 = W1 * x';
    a1 = 1 ./ (1 + exp(-z1));
    z2 = W2 * a1 + b2;
    y_pred = z2';

    prediction = (y_pred .* sigma) + mu;
end

function compare_vectors(name, original, rf, ann)
    fprintf('\n--- %s Comparison (Euclidean distances) ---\n', name);
    fprintf('Original vs RF:  %.4f\n', norm(original - rf));
    fprintf('Original vs ANN: %.4f\n', norm(original - ann));
    fprintf('RF vs ANN:       %.4f\n', norm(rf - ann));
end

function show_image_from_vector(vec, title_str, min_val, max_val)
    s = ceil(sqrt(length(vec)));
    padded = [vec, zeros(1, s^2 - length(vec))];
    img = reshape(padded, [s, s]);

    % Normalize using provided min and max for consistent scaling
    if nargin < 3
        min_val = min(img(:));
        max_val = max(img(:));
    end

    if max_val > min_val
        img_norm = (img - min_val) / (max_val - min_val);
    else
        img_norm = zeros(size(img));
    end

    figure;
    imshow(img_norm);
    title(title_str);
end
