function results = JBAND_HTV(HSI, params)
% 
% HTV Version
%
%    min_{B,A,S,L} ||B||_{HTV} + \lambda_1 ||A||_{2,1} + \lambda_2 ||L||_{1}
%    s.t. D_v(L) = 0
%         ||V - (B + A + S + L)||_{F} <= \epsilon
%         ||S||_{1} <= \alpha
%
% Inputs: 
%     HSI    - Hyperspectral image of size height x width x bands. 
%     params - Structure containing the algorithm parameters: 
%              lambda_1, lambda_2, epsilon, alpha, tol, max_iter, and use_GPU. 
% 
% Outputs: 
%     results.background_part - Estimated background component. 
%     results.anomaly_part    - Estimated anomaly component. 
%     results.sparse_noise    - Estimated sparse noise component. 
%     results.stripe_noise    - Estimated stripe noise component. 
%     results.gaussian_noise  - Estimated Gaussian noise component. 
%     results.iter            - Number of iterations. 
% 
% Note: 
%     If lambda_2 = 0 and alpha = 0, the sparse and stripe noise components 
%     are omitted from the optimization.    
%

lambda1 = params.lambda1;
lambda2 = params.lambda2;
epsilon = params.epsilon;
alpha   = params.alpha;

if ~isfield(params, 'tol')
    tol = 10^-4;
else
    tol = params.tol;
end

if ~isfield(params, 'max_iter')
    max_iter = 10000;
else 
    max_iter = params.max_iter;
end

if ~isfield(params, 'use_GPU')
    use_GPU = 1;
else 
    use_GPU = params.use_GPU;
end


%% Setting functions

Dv   = @(z) z([2:end, end], :, :) - z; 
Dvt  = @(z) cat(1, -z(1, :, :), -z(2:(end-1), :, :) + z(1:(end-2), :, :), z(end-1, :, :));
Dvh  = @(z) cat(4, z([2:end, end], :, :) - z, z(:, [2:end, end], :) - z);
Dvht = @(z) cat(1, -z(1, :, :, 1), -z(2:end-1, :, :, 1) + z(1:end-2, :, :, 1), z(end-1, :, :, 1)) ...
          + cat(2, -z(:, 1, :, 2), -z(:, 2:end-1, :, 2) + z(:, 1:end-2, :, 2), z(:, end-1, :, 2));


%% Initializings variables

[n1, n2, n3] = size(HSI);

if use_GPU == 1
    B  = gpuArray(single(zeros(n1, n2, n3))); % Background part
    A  = gpuArray(single(zeros(n1, n2, n3))); % Anomaly part
    S  = gpuArray(single(zeros(n1, n2, n3))); % Sparse noise
    L  = gpuArray(single(zeros(n1, n2, n3))); % Stripe noise
    Y1 = gpuArray(single(zeros(n1, n2, n3, 2)));
    Y2 = gpuArray(single(zeros(n1, n2, n3)));
    Y3 = gpuArray(single(zeros(n1, n2, n3)));
else
    B  = single(zeros(n1, n2, n3)); % Background part
    A  = single(zeros(n1, n2, n3)); % Anomaly part
    S  = single(zeros(n1, n2, n3)); % Sparse noise
    L  = single(zeros(n1, n2, n3)); % Stripe noise
    Y1 = single(zeros(n1, n2, n3, 2));
    Y2 = single(zeros(n1, n2, n3));
    Y3 = single(zeros(n1, n2, n3));
end


%% Calculating stepsizes

if (lambda2 == 0) && (alpha == 0)
    mode = 2;
    
    gamma_B = single(1/9);
    gamma_A = single(1);
    gamma_Y1 = single(1/2);
    gamma_Y3 = single(1/2);
else 
    mode = 1;

    gamma_B = single(1/9);
    gamma_A = single(1);
    gamma_S = single(1);
    gamma_L = single(1/5);
    gamma_Y1 = single(1/4);
    gamma_Y2 = single(1/4);
    gamma_Y3 = single(1/4);
end


%% Start algorithm

disp('****************** Algorithm starts (JBAND_HTV) ******************')

if (mode == 1)
    for i = 1:max_iter
        
        B_pre = B;
        A_pre = A;
        S_pre = S;
        L_pre = L;
        Y1_pre = Y1;
        Y2_pre = Y2;
        Y3_pre = Y3;

        % Updating B : Step 3
        B = B_pre - gamma_B.*(Dvht(Y1_pre) + Y3_pre);
    
        % Updating A : Step 4
        A = proxL12Norm_band(A_pre - gamma_A.*Y3_pre, gamma_A.*lambda1);
    
        % Updating S : Step 5
        S = projL1Ball(S_pre - gamma_S.*Y3_pre, alpha);
    
        % Updating L : Step 6
        L = proxL1Norm(L_pre - gamma_L.*(Dvt(Y2_pre) + Y3_pre), gamma_L.*lambda2);
    
        % Updating Y1 : Steps 7 and 8
        Y1_tmp = Y1_pre + gamma_Y1.*Dvh(2*B - B_pre);
        Y1 = Y1_tmp - gamma_Y1.*proxL12Norm_band(Y1_tmp./gamma_Y1, 1./gamma_Y1);

        % Updating Y2 : Steps 9 and 10 
        Y2 = Y2_pre + gamma_Y2.*Dv(2*L - L_pre);

        % Updating Y3 : Steps 11 and 12 
        Y3_tmp = Y3 + gamma_Y3.*(2*(B + A + S + L) - (B_pre + A_pre + S_pre + L_pre));
        Y3 = Y3_tmp - gamma_Y3.*projL2Ball(Y3_tmp./gamma_Y3, HSI, epsilon);

        % stopping condition
        error_num = sqrt(sum(((B_pre + A_pre + S_pre + L_pre) - (B + A + S + L)).^2, 'all')/sum((B_pre + A_pre + S_pre + L_pre).^2, 'all'));
        
        iter = i;
        if error_num < tol
            break;
        end
        
        % if mod(i, 100) == 0
        %     disp(['iter = ', num2str(i), ', error_num = ', num2str(error_num)]);
        % end
    end
elseif (mode == 2)
    for i = 1:max_iter

        B_pre = B;
        A_pre = A;
        Y1_pre = Y1;
        Y3_pre = Y3;

        % Updating B : Step 3
        B = B_pre - gamma_B.*(Dvht(Y1_pre) + Y3_pre);
    
        % Updating A : Step 4
        A = proxL12Norm_band(A_pre - gamma_A.*Y3_pre, gamma_A.*lambda1);

        % Updating Y1 : Steps 7 and 8
        Y1_tmp = Y1_pre + gamma_Y1.*Dvh(2*B - B_pre);
        Y1 = Y1_tmp - gamma_Y1.*proxL12Norm_band(Y1_tmp./gamma_Y1, 1./gamma_Y1);

        % Updating Y3 : Steps 11 and 12 
        Y3_tmp = Y3 + gamma_Y3.*(2*(B + A) - (B_pre + A_pre));
        Y3 = Y3_tmp - gamma_Y3.*projL2Ball(Y3_tmp./gamma_Y3, HSI, epsilon);

        % stopping condition
        error_num = sqrt(sum(((B_pre + A_pre) - (B + A)).^2, 'all')/sum((B_pre + A_pre).^2, 'all'));
        
        iter = i;
        if error_num < tol
            break;
        end

    end
end

disp('****************** Algorithm ends (JBAND_HTV) ********************')


results.background_part = gather(B);
results.anomaly_part    = gather(A);
results.sparse_noise    = gather(S);
results.stripe_noise    = gather(L);
results.gaussian_noise  = gather(HSI - (B + A + S + L));
results.iter            = iter;

end