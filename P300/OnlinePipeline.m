function [] = OnlinePipeline()
%OnlinePipeline - this is the online inference pipeline for communication
%

%clear any previous open windows - hopefully.
close all; clear; clc;

%% Load model & record parameters

    %%% Here we allow selecting only a user directory or a specific model directory
    modelFolder = uigetdir('G:\.shortcut-targets-by-id\1EX7NmYYOTBYtpFCqH7TOhhm4mY31oi1O\P300-Recordings', ...
        'Choose Desired Directory for Saving Recordings');

    % if the path of the folder won't contain allModels we assume that only a
    % subject folder was selecte thus we need to select a specific model fodler
    if ~strfind(modelFolder, Utils.Config.modelDirName)
        modelFolder = selectModelByAcc(modelFolder);
    end

    loadedModel = load(strcat(modelFolder, 'model.mat'));
    if stfind(modelFolder,'SVM')
        svmModel = loadedModel.finalModel;
        modelPredictFunc = @(testData) svmModel.predict(testData);
    elseif stfind(modelFolder,'LDA')
        modelPredictFunc = GetLDAPredictionFunc(loadedModel.finalModel);
        error('to do')
    else
        error('Online training unknwon model type')
    end


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

    % This is a heuristic that needs to be changed to allow shorter online phase than oflline training pahse
    if ceil(1/oddBallProb)*numClasses >= 2*triggersInTrial 
        triggersInTrial = ceil(1/oddBallProb)*numClasses + startingNormalTriggers;
    end

    %% Online Recording

    [EEG, trainingVector, triggersTimes, ~, fig, classesNames] = Recording.OnlineTraining(...
                                                    timeBetweenTriggers, calibrationTime, timeBeforeJitter,...
                                                    numClasses, oddBallProb, triggersInTrial, ...
                                                    triggerBankFolder, is_visual, ...
                                                    preTrialPause, maxRandomTimeBetweenTriggers);

    %% predict

    [~, ~, processedEEG] = preprocessing(EEG, triggersTimes, trainingVector, ...
                                                           preTriggerRecTime, triggerWindowTime);

    testData = Models.processedDataTo2dMatrixMeanChannels(processedEEG, trainingVector, 1);
    predictions = modelPredictFunc(testData);

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


function [folderPath] = selectModelByAcc(baseDir)
    modelsDir = [baseDir '\' Utils.Config.modelDirName '\'];
    dirs = ls(modelsDir);
    sortedDirs = sort(dirs);
    sortedDirs = sortedDirs(2:end, :);   % remove . & ..
    bestModelDir = sortedDirs(1:end, :);
    folderPath = [modelsDir bestModelDir '\'];
end