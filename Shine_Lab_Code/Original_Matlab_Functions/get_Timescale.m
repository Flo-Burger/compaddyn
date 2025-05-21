% Timescale
%
% 
%
% ARGUMENTS: data - time x region
%        
%
% OUTPUT:
%       
%
%
%
% TO DO:
%
%      
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function time_sc = get_Timescale_1(data, pp)
    % data: time x region
    % pp: optional, region index. If not provided, average across regions

    [T, N] = size(data);
    acf = zeros(T, N);

    for rr = 1:N
        [acf(:, rr), lags] = autocorr(data(:, rr), 'NumLags', T-1);
    end

    % If pp is not provided or empty, use the average across regions
    if nargin < 2 || isempty(pp)
        acf_used = mean(acf, 2);
    else
        if pp > N || pp < 1
            error('Region index pp is out of bounds.');
        end
        acf_used = acf(:, pp);
    end

    grad_acf = gradient(acf_used);
    x_max = find(grad_acf >= 0, 1, 'first');
    if isempty(x_max)
        x_max = length(acf_used);
    end

    x = lags(1:x_max);
    y = acf_used(1:x_max);

    g = fittype('a - b*exp(-c*x)', 'independent', 'x');
    f0 = fit(x, y, g, 'StartPoint', [[ones(size(x)), -exp(-x)]\y; 1]);
    fitvalues = coeffvalues(f0);
    time_sc = fitvalues(3); % decay constant
end


[timepoints, regions, num_subjects] = size(cort_ts1c);
timescales = zeros(num_subjects, 1);

for s = 1:num_subjects
    data_subj = cort_ts1c(:, :, s); % already time x region
    timescales(s) = get_Timescale_1(data_subj);
end
