% Susceptibility
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

function sus = get_Susceptibility_1(data)

    [Z,~,~] = zscore(data,0,1); % Binarize data
    N = size(data, 2);  
    density = sum(Z > 0,2)/N; % Number of nodes above mean
    sus = (mean(density.^2) - mean(density)^2)/mean(density); % Variance above and below mean

end

T = size(cort_ts1c, 1); % number of time points
R = size(cort_ts1c, 2); % number of regions
S = size(cort_ts1c, 3); % number of subjects

susceptibilities = zeros(S, 1); % Preallocate output vector

for subj = 1:S
    data_subj = cort_ts1c(:,:,subj);              % Extract time x region data for subject
    susceptibilities(subj) = get_Susceptibility_1(data_subj); % Compute susceptibility
end
