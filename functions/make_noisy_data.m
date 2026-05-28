function HSI = make_noisy_data(HSI, params)

    sigma = params.sigma;
    Sp_rate = params.Sp_rate;
    Sl_rate = params.Sl_rate;
    
    if (Sl_rate ~= 0) % add vertical stripe noise
        HSI = addStripe(HSI, Sl_rate);
    end
    
    if (sigma ~= 0) % add white Gaussian noise
        HSI = addGaussian(HSI, sigma);
    end

    if Sp_rate ~= 0 % add salt-and-pepper noise
        HSI = addSparse(HSI, Sp_rate);
    end

end


%% functions

function Y = addStripe(X, Sl_rate)
    [rows, cols, bands] = size(X);
    intensity_stripe = 0.3;
    sigma_stripe = Sl_rate;

    sparse_stripe = 2*(imnoise(0.5*ones(1, cols, bands), "salt & pepper", Sl_rate) - 0.5).*rand(1, cols, bands).*ones(rows, cols, bands);
    warm_stripe = sigma_stripe*randn(1, cols, bands).*ones(rows, cols, bands);
    stripe_noise = warm_stripe + sparse_stripe;
    stripe_noise = intensity_stripe.*stripe_noise./max(abs(stripe_noise), [], "all");
    Y = X + stripe_noise;
end

function Y = addGaussian(X, sigma)
    Y = X + sigma*randn(size(X));
end

function X = addSparse(X, Sp_rate)
    [rows, cols, bands] = size(X);
    sparse_noise = 0.5*ones(rows, cols, bands); 
    sparse_noise = imnoise(sparse_noise, 'salt & pepper', Sp_rate);

    X(sparse_noise == 0) = 0;
    X(sparse_noise == 1) = 1;
end