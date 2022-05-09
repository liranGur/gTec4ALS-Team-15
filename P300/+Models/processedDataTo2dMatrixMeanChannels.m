function [data, targetsVector] = processedDataTo2dMatrixMeanChannels(processedEEG, trainingClasses, ignoreBaseClass)
% processedDataTo2dMatrixMeanChannels - converts the processed EEG data into a LDA
% compatibale data. It is done by averaging on all channels
% 
% INPUTS:
%   - processedEEG - EEG data after preprocessing. shape: #trials, #classe, eegChannels, data size after preprocessing
%   - trainingClasses - vector of target class in each trial
%   - ignoreBaeClass - should ignore call id 1
% 
% OUTPUTS:
%   - data - 2d matrix for LDA. each row is each sample (trial*num classes) and columns are the samples EEG data
%   - targetsVector - vector with 1 & 0 indicating if a sample was also the target of that trial

    processedEEG = squeeze(mean(processedEEG, 3));
    numClasses = length(unique(trainingClasses));
    
    
    numTrials = size(processedEEG,1);
    numClassesInData = size(processedEEG,2);
    sampleSize = size(processedEEG,3);
    
    dataNumRows = (numClasses) * numTrials;
    data = zeros(dataNumRows, sampleSize);  
    targetsVector = zeros(1, dataNumRows); 
    
    for trial=1:size(processedEEG, 1)
        dataRowStartIdx = (trial-1)*numClasses;
        for class=1:numClassesInData
            matrixClsIdx = class;
            if ignoreBaseClass
                if class == 1
                    continue
                else
                    matrixClsIdx = matrixClsIdx - 1;
                end
            end
            
            data(dataRowStartIdx+matrixClsIdx, :) = squeeze(processedEEG(trial, class, :));
            if trainingClasses(trial) == class
                targetsVector(dataRowStartIdx+matrixClsIdx) = 1;
            end
        end
    end

end