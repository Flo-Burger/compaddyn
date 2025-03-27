% Regional Diversity
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

function output = get_Regional_Diversity_1(data,options)

    
    % -------------------- Functional Connectivity
    fC = corr(data(:,:));
    fC(logical(eye(size(fC)))) = 0;

    % -------------------- Regional diversity  --------------
    % Grab the upper triangle of the regional FC
    up_idx  = ones(size(fC));
    up_idx = triu(up_idx,1);
    state_region = fC(logical(up_idx));
    
    % Standard deviation of the regional FC
    output = std(state_region(:));

end

subject1_data = cort_ts1c(:, :, 2);  % Extract time x regions for subject 1
diversity = get_Regional_Diversity_1(subject1_data);
disp(diversity);
