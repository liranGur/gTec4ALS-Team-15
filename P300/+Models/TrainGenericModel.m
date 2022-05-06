function [meanAcc, valAcc, predictions, targets] = TrainGenericModel(modelName, data, labels, numOfFolds)
  
    if strcmp(modelName, 'LDA')
        modelFunc = @LdaModel;
    elseif strcmp(modelName, 'SVM')
        modelFunc = @SvmModel;
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
    testIdx = ((foldNum-1)*foldSize + 1) : (foldSize*foldNum);
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