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

function output = get_Timescale(data,options)

    acf = [];
    for rr = 1: N
        [acf(:,rr),lags] = autocorr(data(:,rr));
    end
    avg_acf = mean(acf,2);
    grad_acf = gradient(avg_acf(:,pp));
    x_max = find(grad_acf >= 0,1,'first');

    x = lags(1:x_max);
    y = avg_acf(1:x_max,pp);
    g = fittype('a-b*exp(-c*x)');
    f0 = fit(x,y,g,'StartPoint',[[ones(size(x)), -exp(-x)]\y; 1]);
    fitvalues = coeffvalues(f0);
    time_sc = fitvalues(3); % save c parameter

end