function [meanAcc, valAcc, predictions, targets, finalModel] = TrainGenericModel(modelName, data, labels, numOfFolds)
% TrainGenericModel - train and evaluate using K-Fold for LDA / SVM model
% 
% INPUTS:
%   - modelName - name of model to train and eval. optiosns: LDA / SVM
%   - data - train data for model
%   - labels - train targest for model
%   - numOfFolds - K folds num
% 
% OUTPUTS:
%   - meanAcc - mean accuracy on validation of each fold
%   - valAcc - struct with size: (1, numOfFolds) - accuracy on validation of each fold
%   - predictions - struct with size: (1, numOfFolds) - predictions on validation of each fold
%   - targets - struct with size: (1, numOfFolds) - targets of validation of each fold
  
    if strcmp(modelName, 'LDA')
        modelFunc = @LdaModel;
        finalModel ={data, labels, 'diagLinear'};
    elseif strcmp(modelName, 'SVM')
        modelFunc = @SvmModel;
        finalModel = fitcsvm(data, labels, 'KernelFunction','rbf');
    else
        error('Unknown Model type received: %s', modelName);
    end
    
    foldSize = round(size(data,1)/numOfFolds);
    valAcc = zeros(1, numOfFolds);
    targets = cell(1,numOfFolds);
    predictions = cell(1,numOfFolds);
    for fold=1:numOfFolds
        [trainData, trainLabels, testData, targets{fold}] = SplitTrainTest(data, labels, fold, foldSize);
        predictions{fold} = modelFunc(trainData, trainLabels, testData);
        valAcc(fold) = sum(predictions{fold} == targets{fold}) / length(targets{fold});
    end
    
    meanAcc = mean(valAcc);
    
end


function [trainData, trainLabels, testData, testLabels] = SplitTrainTest(data, labels, foldNum, foldSize)
    numOfSamples = size(data,1);
    endIdx = (foldSize*foldNum);
    %Last fold will be smaller if numOfSamples/foldSize isn't a round number
    if endIdx > numOfSamples; endIdx = numOfSamples; end
    testIdx = ((foldNum-1)*foldSize + 1) : endIdx;
    trainIdx = setdiff(1:numOfSamples, testIdx);
    trainData = data(trainIdx,:);
    trainLabels = labels(trainIdx);
    testData = data(testIdx,:);
    testLabels = labels(testIdx);
end


function [predictions] = SvmModel(trainData, trainLabels, testData)
    currModel = fitcsvm(trainData,trainLabels, 'KernelFunction','rbf');
    predictions = currModel.predict(testData);
    predictions = reshape(predictions, 1, length(predictions));
end

function [predictions] = LdaModel(trainData, trainLabels, testData)
    [predictions , ~] = classify(testData , trainData, trainLabels , 'diagLinear');
    predictions = reshape(predictions, 1, length(predictions));
end