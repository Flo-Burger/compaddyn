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

function sus = get_Susceptibility(data)

    [Z,~,~] = zscore(data,0,1); % Binarize data
    density = sum(Z > 0,2)/N; % Number of nodes above mean
    sus = (mean(density.^2) - mean(density)^2)/mean(density); % Variance above and below mean

end