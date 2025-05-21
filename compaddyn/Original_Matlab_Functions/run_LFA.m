function [lmse,msd] = run_LFA(data_ts,n_lag, exp_var_lim)
%RUN_LFA Summary of this function goes here
%   Detailed explanation goes here
%
%
% Input:
%   data_ts = 3d matrix(double); variable x time x subj/trial/run
%
%   n_lag = Number of future timepoints to predict (scalar int)
%
%   exp_var_lim = percentage of explained variance to predict (scalar - double)
%

    
    if nargin < 2
        n_lag = 10; % Number of future time-points to predict
    end
    if nargin < 3
        exp_var_lim = 99; % Variance explained
    end
    

    % Data dimensions
    [~,n_time,n_subjs] = size(data_ts);


    for subj = 1: n_subjs
        subj_ts = data_ts(:,:,subj);

        X = subj_ts(:,1:end-1); % X_t
        Y = subj_ts(:,2:end); % X_t+1
        
        % SVD
        [U, S, V]=svd(X,0); % X = U*S*V'

        % ---- Variance Explained
        exp_var = 100.*diag(S).^2./sum(diag(S).^2);
        accum_exp_var = cumsum(exp_var);
        n_pcs = find(accum_exp_var > exp_var_lim,1,'first');

        % Reduce rank
        U=U(:,1:n_pcs); % Space
        V=V(:,1:n_pcs); % Time
        S=S(1:n_pcs,1:n_pcs);

        % Projection timeseries
        X_svd = (S*V.').';

        % Estimate linear propagator A_tilde
        A_tilde=U'*Y*V/S; % Y = AX -> A = Y*inv(X)
        % X = U*S*V'

        % ----- Forecast Linear model
        % For each timepoint
        for ss = 1:n_time-n_lag
            X_plus = X_svd(ss,:); % Initialize current point
            % For each lag into the future
            for ll = 1 : n_lag
                % MSE linear model predictions
                X_plus(ll+1,:) = A_tilde*X_plus(ll,:).';
                lmse(ss,ll,subj) = mean((X_plus(ll,:) - X_svd(ss+ll-1,:)).^2);

                % MSD of autocorrelation
                msd(ss,ll,subj) = mean((X_svd(ss,:) - X_svd(ss+ll-1,:)).^2);
            end
        end
        %disp(subj)
    end
end

