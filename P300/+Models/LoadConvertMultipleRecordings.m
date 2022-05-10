function [data, targets, numFolds] = LoadConvertMultipleRecordings(folderPath, names, forcePreprocess)
% LoadConvertMultipleRecordings - load data from multiple recordings and create data for LDA/SVM models
% 
% INPUTS:
%   - folderPath - base folder where other recordings folder are
%   - names - names of folders to load data from
%   - forcePreprocess - ignore preprocessedEEG data in folder and load raw data and preprocess here
% 
% OUTPUTS:
%   - data - data for training and validation of model
%   - targets - y target for model training size = rows in datt
%   - numOfFolds - = length(names) 

    processedName = 'processedEEG.mat';
      
    for i=1:length(names)
        loadFolder = strcat(folderPath, names{i});
        files = dir(loadFolder);
        hasProcessed = 0;
        
        if forcePreprocess
            for j=1:length(files)
                if strcmp(processedName,files(j).name) == 1
                    hasProcessed = 1;
                    break
                end
            end
        end
        
        if hasProcessed == 1 
            tmpProcessed = load(strcat(loadFolder,'\',processedName));
            allProcessed{i} = tmpProcessed.processedEEG;
        else
            EEG = load(strcat(loadFolder,'\EEG.mat'));
            trainingVec = load(strcat(loadFolder,'\trainingSequences.mat'));
            triggersTimes = load(strcat(loadFolder,'\triggersTime.mat'));
            EEG = EEG.EEG;
            trainingVec = trainingVec.trainingVec;
            triggersTimes = triggersTimes.triggersTimes;
            [~, ~, processedEEG] = preprocessing(EEG, triggersTimes, trainingVec);   
             allProcessed{i} = processedEEG;    
        end
        
        tmpTargets = load(strcat(loadFolder,'\trainingLabels.mat'));
        allTragets{i} = tmpTargets.expectedClasses;
    end
    
    
    for i=1:length(allProcessed)
        [preparedData{i}, preparedTargets{i}] = Models.processedDataTo2dMatrixMeanChannels(allProcessed{i}, allTragets{i}, 1);
    end
    
    data = preparedData{1};
    targets = preparedTargets{1};
    for i=2:length(preparedData)
        data = cat(1,data, preparedData{i});
        targets = cat(2, targets, preparedTargets{i});
    end
    
    numFolds = length(preparedData);
end

