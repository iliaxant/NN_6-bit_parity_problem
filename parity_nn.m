%% 1) Configurations and training dataset creation
% Execution operations are set by the user's configs and the training 
% dataset is created. "Training Set Specifications" are printed as well.

clear;
clc;


% ---------------- User Input -----------------

% Leave blank for random results or set seed for reproducable execution.
rng();

% Set to 1 for training/testing the best examined network.
% Set to 0 for training/testing all the examined networks.
train_mode = 1;

% Set the percentage of dataset used for training (you also need to 
% uncomment the required sections).
data_split = 0.8;

% ---------------------------------------------


digits = 6;

[inputs, targets] = create_dataset(digits);


% % Uncomment for split dataset
% 
% instances = length(inputs);          
% train_num = round(data_split * instances);    
% idx = randperm(instances);           
% 
% train_idx = idx(1:train_num);           
% test_idx = idx(train_num + 1:end);      
% 
% inputs_train = inputs(:, train_idx);   
% targets_train = targets(:, train_idx);
% 
% inputs_test = inputs(:, test_idx);     
% targets_test = targets(:, test_idx);


even_parities = length(find(targets));
odd_parities = length(targets) - even_parities;

[target_vars,~] = size(targets);


fprintf('------- Training Set Specifications -------\n\n');

fprintf("I/O instances: %d\n\n", length(inputs));

fprintf("Number of input variables: %d\n", digits);
fprintf("Number of output variables: %d\n\n", target_vars);

fprintf("Total even parities (Class 1): %d\n", even_parities);
fprintf("Total odd parities (Class 2): %d\n\n", odd_parities);

fprintf('-------------------------------------------\n\n')



%% 2) Network structures and parameters definition
% All the characteristics of multiple Neural Network are defined (num of 
% layers, num of neurons, transfer functions etc). Training hyperparameters 
% are also defined.


nets = cell(1,6);
net_names = cell(1,6);
idx = 1;

% "net_6" is the BEST PERFORMING NN of all the ΝΝs tested.

% ---------------- Neural Network 1 (net_1) -----------------

net_name = 'net_1'; 
net = feedforwardnet([12, 6], 'trainlm');  % Levenberg-Marquardt


% Layer initialization method
net.initFcn = 'initlay';
for i = 1:length(net.layers)
    net.layers{i}.initFcn = 'initnw';  % Nguyen-Widrow initialization
end
net = init(net);


% Transfer/Activation functions
for i = 1:length(net.layers) - 1
    net.layers{i}.transferFcn = 'tansig'; 
end
net.layers{end}.transferFcn = 'purelin';


% Dataset division (it is not used as it defined externally of toolbox)
net.divideFcn = 'dividetrain'; % 100-percent training


% Loss function
net.performFcn = 'mse';  % Mean Squared Error


% Training hyperparameters
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-7;
net.trainParam.show = 50;
net.trainParam.showWindow = true;


nets{idx} = net;
net_names{idx} = net_name;
idx = idx + 1;

% -----------------------------------------------------------




% ---------------- Neural Network 2 (net_2) -----------------

net_name = 'net_2';
net = feedforwardnet([20], 'trainlm');  % Levenberg-Marquardt


% Layer initialization method
net.initFcn = 'initlay';
for i = 1:length(net.layers)
    net.layers{i}.initFcn = 'initnw';  % Nguyen-Widrow initialization
end
net = init(net);


% Transfer/Activation functions
for i = 1:length(net.layers) - 1
    net.layers{i}.transferFcn = 'tansig'; 
end
net.layers{end}.transferFcn = 'purelin';


% Dataset division (it is not used as it defined externally of toolbox)
net.divideFcn = 'dividetrain';  % 100-percent training


% Loss function
net.performFcn = 'mse';  % Mean Squared Error


% Training hyperparameters
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-7;
net.trainParam.show = 50;
net.trainParam.showWindow = true;


nets{idx} = net;
net_names{idx} = net_name;
idx = idx + 1;

% -----------------------------------------------------------




% ---------------- Neural Network 3 (net_3) -----------------

net_name = 'net_3';
net = feedforwardnet([20], 'trainrp');  % Resilient Backpropagation

% Layer initialization method -> Zero weight and bias
for i = 1:numel(net.IW)
    net.IW{i} = zeros(size(net.IW{i}));
end
for i = 1:numel(net.LW)
    net.LW{i} = zeros(size(net.LW{i}));
end
for i = 1:numel(net.b)
    net.b{i} = zeros(size(net.b{i}));
end


% Transfer/Activation functions
for i = 1:length(net.layers) - 1
    net.layers{i}.transferFcn = 'tansig'; 
end
net.layers{end}.transferFcn = 'purelin';


% Dataset division (it is not used as it defined externally of toolbox)
net.divideFcn = 'dividetrain';  % 100-percent training


% Loss function
net.performFcn = 'mse';  % Mean Squared Error


% Training hyperparameters
net.trainParam.epochs = 5000;
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-7;
net.trainParam.show = 50;
net.trainParam.showWindow = true;


nets{idx} = net;
net_names{idx} = net_name;
idx = idx + 1;

% -----------------------------------------------------------




% ---------------- Neural Network 4 (net_4) -----------------

net_name = 'net_4';
net = feedforwardnet([20], 'trainscg');  % Scaled Conjugate Gradient


% Layer initialization method
net.initFcn = 'initlay';
for i = 1:length(net.layers)
    net.layers{i}.initFcn = 'initnw';  % Nguyen-Widrow initialization
end
net = init(net);


% Transfer/Activation functions
for i = 1:length(net.layers) - 1
    net.layers{i}.transferFcn = 'tansig'; 
end
net.layers{end}.transferFcn = 'purelin';


% Dataset division (it is not used as it defined externally of toolbox)
net.divideFcn = 'dividetrain';  % 100-percent training


% Loss function
net.performFcn = 'mse';  % Mean Squared Error


% Training hyperparameters
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-7;
net.trainParam.show = 50;
net.trainParam.showWindow = true;


nets{idx} = net;
net_names{idx} = net_name;
idx = idx + 1;

% -----------------------------------------------------------




% ---------------- Neural Network 5 (net_5) -----------------

net_name = 'net_5';
net = feedforwardnet([10], 'trainlm'); % Levenberg-Marquardt


% Layer initialization method -> Zero weight and bias
for i = 1:numel(net.IW)
    net.IW{i} = zeros(size(net.IW{i}));
end
for i = 1:numel(net.LW)
    net.LW{i} = zeros(size(net.LW{i}));
end
for i = 1:numel(net.b)
    net.b{i} = zeros(size(net.b{i}));
end


% Transfer/Activation functions
for i = 1:length(net.layers) - 1
    net.layers{i}.transferFcn = 'tansig'; 
end
net.layers{end}.transferFcn = 'purelin';


% Dataset division (it is not used as it defined externally of toolbox)
net.divideFcn = 'dividetrain';  % 100-percent training


% Loss function
net.performFcn = 'mse';  % Mean Squared Error


% Training hyperparameters
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-7;
net.trainParam.show = 50;
net.trainParam.showWindow = true;


nets{idx} = net;
net_names{idx} = net_name;
idx = idx + 1;

% -----------------------------------------------------------




% ----------- (BEST NN) Neural Network 6 (net_6) ------------

net_name = 'net_6';
net = feedforwardnet([20], 'trainlm'); % Levenberg-Marquardt


% Layer initialization method -> Zero weight and bias
for i = 1:numel(net.IW)
    net.IW{i} = zeros(size(net.IW{i}));
end
for i = 1:numel(net.LW)
    net.LW{i} = zeros(size(net.LW{i}));
end
for i = 1:numel(net.b)
    net.b{i} = zeros(size(net.b{i}));
end


% Transfer/Activation functions
for i = 1:length(net.layers) - 1
    net.layers{i}.transferFcn = 'tansig'; 
end
net.layers{end}.transferFcn = 'purelin';


% Dataset division (it is not used as it defined externally of toolbox)
net.divideFcn = 'dividetrain';  % 100-percent training


% Loss function
net.performFcn = 'mse';  % Mean Squared Error


% Training hyperparameters
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-7;
net.trainParam.show = 50;
net.trainParam.showWindow = true;


nets{idx} = net;
net_names{idx} = net_name;
idx = idx + 1;

% -----------------------------------------------------------



%% 3) Neural Network training 
% Training of neural network is executed. Useful information regarding the 
% excecution are printed. Results from the GUI are also visible (only of 
% the last trained network).



tr_all = cell(1, length(nets));
if train_mode
    
    % Set below the index of the Network you want to keep for mode 1. Set
    % equal to 6 to view "net_6" which is the BEST PERFORMING NN of all the
    % ΝΝs tested.
    idx = 6; 
    
    
    % ---------------- Neural Network 5 (net_5) -----------------
    
    net = nets{idx};
    net_name = net_names{idx};
    
    
    % % Comment the FIRST line for data split and the SECOND for a not 
        % % split dataset.
        [net, tr] = train(net, inputs, targets);
%         [net, tr] = train(net, inputs_train, targets_train);


    nets{idx} = net; 

    fprintf('\n--------- "%s" Training Details --------\n\n', net_name);

    fprintf('Epochs: %d / %d  (Total time: %s)\n', tr.epoch(end), ...
        net.trainParam.epochs, time_calc(tr.time(end)));
    fprintf('Termination: %s\n', tr.stop);
    fprintf('Best performance at epoch: %d\n', tr.best_epoch);
    fprintf('Loss function value at best Epoch: %.10f\n\n', tr.perf(tr.best_epoch));

    fprintf('-------------------------------------------\n\n');

    tr_all{idx} = tr;
    
    % -----------------------------------------------------------
    
else
    
    for i = 1:length(nets)
        
        net = nets{i};
        net_name = net_names{i};
        
        
        % % Comment the FIRST line for data split and the SECOND for a not 
        % % split dataset.
        [net, tr] = train(net, inputs, targets);
%         [net, tr] = train(net, inputs_train, targets_train);
        

        nets{i} = net; 

        fprintf('\n--------- "%s" Training Details --------\n\n', net_name);

        fprintf('Epochs: %d / %d  (Total time: %s)\n', tr.epoch(end), ...
            net.trainParam.epochs, time_calc(tr.time(end)));
        fprintf('Termination: %s\n', tr.stop);
        fprintf('Best performance at epoch: %d\n', tr.best_epoch);
        fprintf('Loss function value at best Epoch: %.10f\n\n', tr.perf(tr.best_epoch));

        fprintf('-------------------------------------------\n\n');

        tr_all{i} = tr;

    end
    
end




%% 4) Model Testing
% The trained models are evaluated. Useful evaluation messages 
% (accuracy, etc.)as well as graphs are printed.



% % Uncomment both lines for a split dataset
% inputs = inputs_test;
% targets= targets_test;


if train_mode
    
    net = nets{idx};
    net_name = net_names{idx};
    tr = tr_all{idx};

    y = net(inputs);           
    y_pred = max(0, min(1, y)); 
    y_pred = round(y_pred);           
    correct = sum(y_pred == targets);  
    accuracy = correct / length(targets);  

    fprintf('\n------------- "%s" Model Evaluation ------------\n\n', net_name);

    fprintf('Raw outputs: [%s]\n', sprintf('%.4f ', y));
    fprintf('Correct predictions: %d / %d\n', correct, length(targets));
    fprintf('Accuracy: %.2f%%\n\n', accuracy * 100);

    fprintf('---------------------------------------------------\n\n');

    figure(1);
    pred_heatmap(inputs, y_pred, targets, net_name)

    figure(2);
    plotconfusion([targets;1-targets],[y;1-y]);
    set(gcf, 'Position', [300, 300, 400, 400]);
    title(['Confusion Matrix for "' net_name '" model']);

    figure(3);
    plotperform(tr);
    set(gcf, 'Position', [800, 300, 450, 450]);
    
else
    
    for i = 1:length(nets)
    
        net = nets{i};
        net_name = net_names{i};
        tr = tr_all{i};

        y = net(inputs);
        y_pred = max(0, min(1, y));
        y_pred = round(y_pred);            
        correct = sum(y_pred == targets);  
        accuracy = correct / length(targets);
        
        fprintf('\n------------- "%s" Model Evaluation ------------\n\n', net_name);

        fprintf('Raw outputs: [%s]\n', sprintf('%.4f ', y));
        fprintf('Correct predictions: %d / %d\n', correct, length(targets));
        fprintf('Accuracy: %.2f%%\n\n', accuracy * 100);

        fprintf('---------------------------------------------------\n\n');

        figure(1);
        pred_heatmap(inputs, y_pred, targets, net_name)

        figure(2);
        plotconfusion([targets;1-targets],[y;1-y]);
        set(gcf, 'Position', [300, 300, 400, 400]);
        title(['Confusion Matrix for "' net_name '" model']);

        figure(3);
        plotperform(tr);
        set(gcf, 'Position', [800, 300, 450, 450]);

        fprintf('\nPress any key to continue...\n\n');
        pause;

    end
    
end

fprintf('\nExecution complete!\n\n');
