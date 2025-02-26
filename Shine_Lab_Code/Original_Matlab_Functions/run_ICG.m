function [activityICG,outPairID] =  ICGLean_1(allData)

%% version 19/10/21 Commented

%Input
%allData -  neuronal timeseries data (rows - neurons/ cols - time)

%ouput
%activityICG - ICG activity for level k
%outPairID - the ids of original neurons grouped at each level


%calc how many rng iterations I can do 
ICGsteps = nextpow2(size(allData,1));

%Catch case if more neurons than a power of 2
if size(allData,1) == 2^ICGsteps
    ICGsteps = ICGsteps + 1;
end


%The first level is the raw activity
activityICG = cell(1,ICGsteps);
activityICG{1} = allData;

%Cell ids correspond to order inputted
outPairID = cell(1,ICGsteps);
outPairID{1}(1:size(allData,1),1) = 1:size(allData,1);

clearvars allData
%% Start the ICG process


for ICGlevel = 2:ICGsteps
    %ICGlevel
    
    %Grab data
    ICGAct = activityICG{ICGlevel-1};
    nData = size(ICGAct,1);
    
    
    %Calculate correlation matrix
%     tic
    rho = corr(ICGAct');
%     toc
    
    C = triu(rho,1);
    clearvars rho
    
      
    %Grab just the upper triangle
    %Find the indices of each element
    upTriIndx = find(triu(true(size(C)),1)>0);
    [allRIndx, allCIndx] = ind2sub(size(C),upTriIndx);
    
    %now it is a vector of correlation values
    C = C(upTriIndx);
    clearvars upTriIndx
    
   %Sort the correlation matrix
    [~,sCI] = sort(C,'descend');

    [~, sCI] = sort(C, 'descend');

    %     %resort row/col indices and turn into pairings 
    allRowIndx = allRIndx(sCI);
    allColIndx = allCIndx(sCI);
    clearvars allRIndx allCIndx C sCI
    
   
    %How many pairs can be made
    numPairsTotal = floor(nData/2);
    
    
    % I also know the size of my output data = half data by time
    %outdat = nan(1,size(RngAct,2));
    outdat = nan(numPairsTotal,size(ICGAct,2));
    
    %to save pairings
    outPair = nan(numPairsTotal,2);
    
    %this is 2 times num pairings before
    outPairID{ICGlevel} = nan(numPairsTotal,2^(ICGlevel-1));

    %% Index counter (optimised code)
    k = 1;
    gdIndex = true(numel(allRowIndx),1);
  
    %tic
    for numPairCnt = 1:numPairsTotal
         
        %Text counter
        % if ~mod(numPairCnt,250)
        %     [numPairCnt numPairsTotal];
        %     100*numPairCnt/numPairsTotal;

        % end
        
        %Grabbing data
        %top row index
        rowNew = allRowIndx(k);
        %grab corresponding data
        datRow = ICGAct(rowNew,:);
        
        %top col index
        colNew = allColIndx(k);
        %grab correspondingdata
        datCol = ICGAct(colNew,:);
        
        
        %ICG process
        %Sum the data
        SpikeAct = datRow+datCol;
        
        % Save data
        outdat(numPairCnt,:) = SpikeAct;
        outPair(numPairCnt,:) = [rowNew colNew];
        
        %Update the list of original pairs
        tempPairs = outPairID{ICGlevel-1}([rowNew colNew],:);
        outPairID{ICGlevel}(numPairCnt,:) = tempPairs(:);
        
        %Update list of available neurons to pair (optimised) 
        gdIndex = allRowIndx ~= rowNew & allRowIndx ~= colNew & allColIndx ~= colNew & allColIndx ~= rowNew & gdIndex; 
        
        %Find the next pairing (greedily)
        k = find(gdIndex,1,'first');
%         knew = find(gdIndex(k+1:end),1,'first');
%         k = knew+k;
    
        
    end
    %toc
    
    
    %Save all the activity
    activityICG{ICGlevel} = outdat;
    clearvars outdat outPair
    
end



end


subject_idx = 1; % Select subject 1 (change this for other subjects)
single_subject_data = fmri_data(:,:,subject_idx);
% % % 
% % % % Suppose your data is in a 2D matrix 'single_subject_data' (neurons x time).
% % % % And your function is ICGLean_1(single_subject_data).
% % % 
tic;  % Start timer
[activityICG, outPairID] = ICGLean_1(single_subject_data);
elapsed_time = toc;  % End timer and capture elapsed seconds
fprintf('MATLAB ICG completed in %.4f seconds.\n', elapsed_time);
% % % 

% [numNeurons, numTimepoints, numSubjects] = size(fmri_data);
% 
% % Initialize timing
% tic;
% activityICG_all = cell(1, numSubjects);
% outPairID_all = cell(1, numSubjects);
% 
% % Run ICG for each subject
% for subj = 1:numSubjects
%     % fprintf('Running ICG for subject %d/%d...\n', subj, numSubjects);
% 
%     single_subject_data = fmri_data(:, :, subj);
%     [activityICG_all{subj}, outPairID_all{subj}] = ICGLean_1(single_subject_data);
% end
% 
% matlab_time = toc;
% fprintf('MATLAB ICG completed for all %d subjects in %.4f seconds.\n', numSubjects, matlab_time);
% 
