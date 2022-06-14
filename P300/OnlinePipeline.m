function [] = OnlinePipeline()
%OnlinePipeline - this is the online inference pipeline for communication
%

%clear any previous open windows - hopefully.
close all; clear; clc;

%% Select Folder & Set Parameters

    %%% Here we allow selecting only a user directory or a specific model directory
    modelFolder = GUIFiles.OnlineModelSelection('G:\.shortcut-targets-by-id\1EX7NmYYOTBYtpFCqH7TOhhm4mY31oi1O\P300-Recordings');

    % if the path of the folder won't contain allModels we assume that only a
    % subject folder was selecte thus we need to select a specific model fodler
    splitPath = strsplit(modelFolder, '\');
    finalFolder = splitPath{length(splitPath)};
    if strcmp(finalFolder, Utils.Config.modelDirName)
        modelFolder = selectModelByAcc(modelFolder);
    end
    
    display(modelFolder);

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

    %% Set up recording
    [eegSampleSize, recordingBuffer, trainingSamples, diffTrigger, classesNames, activateTrigger, fig] = ...
            Recording.RecordingSetup(timeBetweenTriggers, calibrationTime, triggersInTrial, triggerBankFolder, timeBeforeJitter, is_visual);
        
    %% Recording loop
    
    inferenceIdx = 1;
    while 1
        [EEG, trainingVector, triggersTimes, ~, fig, classesNames] = ...
            Recording.OnlineTraining(numClasses, oddBallProb, triggersInTrial, is_visual, ...
                                     maxRandomTimeBetweenTriggers, preTrialPause, timeBetweenTriggers, ...
                                     eegSampleSize, recordingBuffer, trainingSamples, diffTrigger, ...
                                     classesNames, activateTrigger, fig);

        %% Preprocess
        [~, ~, ~, processedEEG] = preprocessing(EEG, triggersTimes, trainingVector, ...
                                                               preTriggerRecTime, triggerWindowTime, ...
                                                               downSampleRate);
        inferenceFile = ['rec_data_',int2str(inferenceIdx),'.mat'];
        save(strcat(modelFolder, inferenceFile), 'processedEEG');
        
        %% Predict
        pythonCommand = ['python .\PythonCode\OnlineInference.py' ' ' modelFolder ' ' inferenceFile];
        [inferedClass, pyOutput] = system(pythonCommand, '-echo');
        
        if inferedClass == -1
            set(fig, 'color', 'black');          % imshow removes background color, therefore we need to set it again before showing more text
            Recording.DisplayTextOnFig('The model was unable to select an answer please try again');
        elseif inferedClass == 1    % This assumes there is an idle class TODO fix this
            set(fig, 'color', 'black');
            Recording.DisplayTextOnFig('<html>The model was unable to select an answer.<br>Probably some problem running the model<br>Program will close now');
            error('Python script of online inference returned 1, there is probably some bug')
        else
            set(fig, 'color', 'black');          % imshow removes background color, therefore we need to set it again before showing more text
            Recording.DisplayTextOnFig(['The selected response was ' classesNames(inferedClass)]);
        end

        GUIFiles.SuspendRun();
        inferenceIdx = inferenceIdx + 1;
    end
                                                   
end


function [folderPath] = selectModelByAcc(modelsDir)
    dirs = dir(modelsDir);
    dirs = {dirs.name};
    dirs = {dirs{3:end}};       % remove . & ..
    sortedDirs = sort(dirs);
    bestModelDir = sortedDirs{length(sortedDirs)};
    folderPath = [modelsDir '\' bestModelDir '\'];
end