function [] = OnlinePipeline()
%OnlinePipeline - this is the online inference pipeline for communication
%

%clear any previous open windows - hopefully.
close all; clear; clc;

%% Load model & record parameters

    %%% Here we allow selecting only a user directory or a specific model directory
    modelFolder = GUIFiles.OnlineModelSelection('G:\.shortcut-targets-by-id\1EX7NmYYOTBYtpFCqH7TOhhm4mY31oi1O\P300-Recordings');

    % if the path of the folder won't contain allModels we assume that only a
    % subject folder was selecte thus we need to select a specific model fodler
    splitPath = strsplit(modelFolder, '\');
    finalFolder = splitPath{length(splitPath)};
    if strcmp(finalFolder, Utils.Config.modelDirName)
        modelFolder = selectModelByAcc(modelFolder);
    end

%     loadedModel = load(strcat(modelFolder, 'model.mat'));
%     if stfind(modelFolder,'SVM')
%         svmModel = loadedModel.finalModel;
%         modelPredictFunc = @(testData) svmModel.predict(testData);
%     elseif stfind(modelFolder,'LDA')
%         modelPredictFunc = GetLDAPredictionFunc(loadedModel.finalModel);
%         error('to do')
%     else
%         error('Online training unknwon model type')
%     end


    %% online parameters

    offlineParameters = load(strcat(modelFolder, 'parameters.mat'));
    offlineParameters = offlineParameters.parametersToSave;

    numClasses = offlineParameters.numClasses;
    timeBetweenTriggers = offlineParameters.timeBetweenTriggers;
    timeBeforeJitter = offlineParameters.beforeJitterTime;
    startingNormalTriggers = offlineParameters.startingNormalTriggers;
    oddBallProb = offlineParameters.oddBallProb;
    triggersInTrial = offlineParameters.triggersInTrial;
    triggerBankFolder = offlineParameters.triggerBankFolder;
    is_visual = offlineParameters.is_visual;
    maxRandomTimeBetweenTriggers = offlineParameters.maxRandomTimeBetweenTriggers;
    preTrialPause = offlineParameters.preTrialPause;
    triggerWindowTime = offlineParameters.triggerWindowTime;
    preTriggerRecTime = offlineParameters.preTriggerRecTime;
    calibrationTime = offlineParameters.calibrationTime;
    downSampleRate = offlineParameters.downSampleRate;

    % This is a stupid way that needs to be changed to allow shorter online phase than oflline training pahse
    if ceil(1/oddBallProb)*numClasses >= 2*triggersInTrial 
        triggersInTrial = ceil(1/oddBallProb)*numClasses + startingNormalTriggers;
    end

    %% Online Recording

    [EEG, trainingVector, triggersTimes, ~, fig, classesNames] = Recording.OnlineTraining(...
                                                    timeBetweenTriggers, calibrationTime, timeBeforeJitter,...
                                                    numClasses, oddBallProb, triggersInTrial, ...
                                                    triggerBankFolder, is_visual, ...
                                                    preTrialPause, maxRandomTimeBetweenTriggers);

    %% Preprocess

    [~, ~, processedEEG] = preprocessing(EEG, triggersTimes, trainingVector, ...
                                                           preTriggerRecTime, triggerWindowTime, ...
                                                           downSampleRate);
    %% Predict
%     testData = Models.processedDataTo2dMatrixMeanChannels(processedEEG, trainingVector, 1);
%     predictions = modelPredictFunc(testData);

    if sum(predictions) ~= 1
        set(fig, 'color', 'black');          % imshow removes background color, therefore we need to set it again before showing more text
        Recording.DisplayTextOnFig('The model was unable to select an answer please try again');
    else
        predictionClass  = find(predictions == 1);
        set(fig, 'color', 'black');          % imshow removes background color, therefore we need to set it again before showing more text
        Recording.DisplayTextOnFig(['The selected response was ' classesNames(predictionClass)]);
    end
    
    GUIFiles.SuspendRun();
                                                   
end


function [func] = GetLDAPredictionFunc(ldaStruct)
    trainData = ldaStruct{1};
    trainLabels = ldaStruct{2};
    type = ldaStruct{3};
    func = @(testData) classify(testData , trainData, trainLabels , type);
end


function [folderPath] = selectModelByAcc(modelsDir)
    dirs = dir(modelsDir);
    dirs = {dirs.name};
    dirs = {dirs{3:end}};       % remove . & ..
    sortedDirs = sort(dirs);
    bestModelDir = sortedDirs{length(sortedDirs)};
    folderPath = [modelsDir '\' bestModelDir '\'];
end