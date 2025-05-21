% Run Energy Landscape

%% MSD Landscape
ndt = 100;               % Number of lags
ds = 0:1:20;             % Number of MSD divisions
nrg_sig = nan(ndt, numel(ds), size(bold_ts, 3));  % Preallocate energy array

for mm = 1:size(bold_ts, 3)
    for tt = 1:ndt
        % Normalize data
        cort_sig = squeeze(zscore(bold_ts(:, :, mm)));
        
        % MSD calculation
        MSD = mean((cort_sig(1 + tt:end, :) - cort_sig(1:end - tt, :)).^2, 2);
        
        % Calculate probability distribution and energy for each dt
        nrg_sig_dt = PdistGaussKern(MSD, ds);
        
        % Pool results across time
        nrg_sig(tt, :, mm) = nrg_sig_dt;
    end
end

%% Function: PdistGaussKern
function [nrg] = PdistGaussKern(dat, ds, bandwidth)
    % Calculates energy using a Gaussian kernel density estimate
    % Inputs:
    %   dat       - MSD values
    %   ds        - Range for probability distribution
    %   bandwidth - Kernel bandwidth (default: 1)
    % Outputs:
    %   nrg       - Energy values

    if nargin < 3
        bandwidth = 1;  % Default bandwidth
    end
    
    % Fit a Gaussian kernel to the data
    pd = fitdist(dat, 'Kernel', 'BandWidth', bandwidth);
    yLC = pdf(pd, ds);  % Probability density function
    
    % Compute energy
    nrg = -1 .* log(yLC);
end
