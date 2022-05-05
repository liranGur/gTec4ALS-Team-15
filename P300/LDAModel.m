function [accuarcy, valAcc, trainErrorRate] = LDAModel(processedEEG, trainingClasses, ignoreBaseClass, ...
                                                       numOfFolds)
% 
% 
% INPUT:
%   - proccesedEEG - EEG signal after all preprocessing and mean. The classes are ordered
%                    shape: shape: #trials, #classes, eegChannels, downsampled window size
%   - targetClasses - the class the users focused on. shape: #trials

    
    
    [data, labels] = prepareData(processedEEG, trainingClasses, ignoreBaseClass);
    
    %K Fold
    foldSize = round(size(data,1)/numOfFolds);
    valAcc = zeros(foldSize);
    trainErrorRate = zeros(foldSize);
    
    for fold=1:numOfFolds
        [trainData, trainLabels, testData, testLabels] = splitTrainTest(data, labels, fold, foldSize);
        [predictions , trainErrorRate(fold)] = classify(testData , trainData, trainLabels , 'linear');
        predictions = reshape(predictions, 1, length(predictions));
        valAcc(fold) = sum(predictions == testLabels) / length(testLabels);
    end
    
    [finalPreds, ~] = classify(data, data, labels, 'linear');
    finalPreds = reshape(finalPreds, 1, length(finalPreds));
    accuarcy = sum(finalPreds == labels) / length(labels);
    
end

function [trainData, trainLabels, testData, testLabels] = splitTrainTest(data, labels, foldNum, foldSize)
    numOfSamples = size(data,1);
    testIdx = ((foldNum-1)*foldSize + 1) : (foldSize*foldNum);
    trainIdx = setdiff(1:numOfSamples, testIdx);
    trainData = data(trainIdx,:);
    trainLabels = labels(trainIdx);
    testData = data(testIdx,:);
    testLabels = labels(testIdx);
end



function [data, targetsVector] = prepareData(processedEEG, trainingClasses, ignoreBaseClass)
% prepareData - This data converts the processed EEG data into a LDA
% compatibale datat. It is done by averaging on all channels

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