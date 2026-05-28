%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the demo file of the method proposed in the following reference:
% 
% K. Sato and S. Ono
% "Joint Background–Anomaly–Noise Decomposition for Robust Hyperspectral 
% Anomaly Detection via Constrained Convex Optimization"
% 
%
% Update history:
% May 28, 2026: v1.0 
%
% Copyright (c) 2026 Koyo Sato and Shunsuke Ono
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;


%% Adding path

gpath_datasets  = genpath('./datasets'); 
gpath_functions = genpath('./functions');
addpath(gpath_datasets);
addpath(gpath_functions);


%% Loading data

load("abu-beach-4.mat");

HSI = normalize_01(data);
ground_truth = double(map);


%% Setting parameters

% size
[rows, cols, bands] = size(HSI);

% use GPU or not
use_GPU = 1; % 0 if you do not use GPU, 1 if you use GPU

% noise
% Case 1: original dataset
sigma   = 0; % standard deviation of Gaussian noise
Sp_rate = 0; % ratio of salt-and-pepper noise
Sl_rate = 0; % ratio of stripe noise

% % Case 2: Gaussian noise
% sigma   = 0.03;
% Sp_rate = 0;
% Sl_rate = 0;
% % Case 3: non-Gaussian noise
% sigma   = 0;
% Sp_rate = 0.03;
% Sl_rate = 0.03;
% % Case 4: mixed noise
% sigma   = 0.01;
% Sp_rate = 0.01;
% Sl_rate = 0.01;
% % Case 5: mixed noise
% sigma   = 0.05;
% Sp_rate = 0.05;
% Sl_rate = 0.05;

% Designs of Background Characterization Function
type_DBCF = 'HTV'; % Recommended
% type_DBCF = 'SSTV';
% type_DBCF = 'HSSTV';
% type_DBCF = 'Nuclear';


%% Generating noisy HS image

noise_settings.sigma = sigma;
noise_settings.Sp_rate = Sp_rate;
noise_settings.Sl_rate = Sl_rate;

HSI = make_noisy_data(HSI, noise_settings);


%% Preparing parameters

% Hyperparameters in the objective function
% Please adjust these parameters depending on the dataset and noise level
lambda1 = 0.75; 
lambda2 = 0; % Set to 0 when no stripe noise is assumed

% parameters for noise
eta     = 0.9;
epsilon = eta*sigma*sqrt(rows*cols*bands*(1 - Sp_rate));
alpha   = eta*0.5*Sp_rate*rows*cols*bands;


%% Hyperspectral Anomaly Detection

params.lambda1 = lambda1;
params.lambda2 = lambda2;
params.epsilon = epsilon;
params.alpha   = alpha;
params.use_GPU = use_GPU;

if strcmp(type_DBCF, 'HTV')
    results = JBAND_HTV(HSI, params);
elseif strcmp(type_DBCF, 'SSTV')
    results = JBAND_SSTV(HSI, params);
elseif strcmp(type_DBCF, 'HSSTV')
    results = JBAND_HSSTV(HSI, params);
elseif strcmp(type_DBCF, 'Nuclear')
    results = JBAND_Nuclear(HSI, params);
else
    error('The selected background characterization function is not supported.');
end

% Generate a detection map
detection_map = sqrt(sum(results.anomaly_part.^2, 3));


%% Calculating metrics

results_AUCs = computeAUCs(detection_map, ground_truth);


%% Displaying metrics

fprintf('AUC_{(P_D, P_F)}   = %.4f\n', results_AUCs.AUC);
fprintf('AUC_{(P_D, tau)}   = %.4f\n', results_AUCs.ATPR);
fprintf('AUC_{(P_F, tau)}   = %.4f\n', results_AUCs.AFPR);


%% Plotting results

detection_map = normalize_01(detection_map);
imshow(detection_map);

rmpath(gpath_datasets);
rmpath(gpath_functions);



%% Functions

function result = normalize_01(data)
    result = (data - min(data(:))) / (max(data(:)) - min(data(:)));
end

function results = computeAUCs(detection_map, ground_truth)
    if (isa(detection_map, 'gpuArray'))
        detection_map = gather(detection_map);
    end
    
    dm = detection_map(:);
    if (max(dm) - min(dm) == 0)
        dm = zeros(size(detection_map));
        dm = dm(:);
    else
        dm = (dm - min(dm)) / (max(dm) - min(dm)); 
    end
    gt = ground_truth(:);
    
    [PF, PD, Thre, AUC] = perfcurve(gt, dm, 1);
    AFPR = sum((Thre(1:end-1)-Thre(2:end)).*(PF(2:end)+PF(1:end-1))/2); 
    ATPR = sum((Thre(1:end-1)-Thre(2:end)).*(PD(2:end)+PD(1:end-1))/2); 

    results.AUC  = AUC;
    results.AFPR = AFPR;
    results.ATPR = ATPR;
end


%% Code ends here.