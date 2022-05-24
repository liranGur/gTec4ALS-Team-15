function [] = OfflinePipline()
% OfflinePipline - This functions starts the P300 offline training pipeline
% recording, preprocessing and creating a model

%clear any previous open windows - hopefully.
close all; clear; clc;

baseFolder = uigetdir('G:\.shortcut-targets-by-id\1EX7NmYYOTBYtpFCqH7TOhhm4mY31oi1O\P300-Recordings', ...
    'Choose Desired Directory for Saving Recordings');

%% Recording Parameter Setting
[is_visual, triggersInTrial, numClasses, subId, numTrials, timeBetweenTriggers, oddBallProb, ...
    calibrationTime, pauseBetweenTrials, triggerBankFolder, beforeJitterTime] = GUIFiles.ParametersGui();


recordingFolder = [baseFolder '\' int2str(subId) '\' strrep(datestr(now), ':','-') '\'];
mkdir(recordingFolder);

%% Training
[EEG, trainingVector, trainingLabels, triggersTimes, backupTimes] = ...
    OfflineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials, numTrials, beforeJitterTime, ...
                    numClasses, oddBallProb, triggersInTrial, ...
                    triggerBankFolder, is_visual);

save(strcat(recordingFolder, 'trainingVector.mat'), 'trainingVector');
save(strcat(recordingFolder, 'EEG.mat'), 'EEG');
save(strcat(recordingFolder, 'trainingLabels.mat'), 'trainingLabels');
save(strcat(recordingFolder, 'triggersTimes.mat'), 'triggersTimes');
save(strcat(recordingFolder, 'backupTimes.mat'), 'backupTimes');

parametersToSave = struct('timeBetweenTriggers', timeBetweenTriggers, ...
                           'calibrationTime', calibrationTime, ...
                           'pauseBetweenTrials',pauseBetweenTrials, ...
                           'numTrials', numTrials, ...
                           'beforeJitterTime', beforeJitterTime, ...
                           'startingNormalTriggers', Utils.Config.startingNormalTriggers, ...
                           'numClasses', numClasses, ...
                           'oddBallProb', oddBallProb, ...
                           'triggersInTrial', triggersInTrial, ...
                           'triggerBankFolder', triggerBankFolder, ...
                           'mode', is_visual, ...
                           'Hz', Utils.Config.Hz, ....
                           'highLim', Utils.Config.highLim, ...
                           'lowLim', Utils.Config.lowLim, ...
                           'pauseBeforeDump', Utils.Config.pauseBeforeDump, ...
                           'preTriggerRecTime', Utils.Config.preTriggerRecTime, ...
                           'triggerWindowTime', Utils.Config.triggerWindowTime);
                           
save(strcat(recordingFolder, 'parameters.mat'), 'parametersToSave');


%% PreProcessing

[splitEEG, meanTriggers, processedEEG] = preprocessing(EEG, triggersTimes, trainingVector);

save(strcat(recordingFolder, 'splitEEG.mat'), 'splitEEG');
save(strcat(recordingFolder, 'meanTriggers.mat'), 'meanTriggers');
save(strcat(recordingFolder, 'processedEEG.mat'), 'processedEEG');

%% Models
[data, targets] = Models.processedDataTo2dMatrixMeanChannels(processedEEG, trainingLabels, 1);
[meanAcc, valAcc, predictions, targets] = Models.TrainGenericModel('SVM', data, targets, 3);

save(strcat(recordingFolder, 'meanAcc.mat'), 'meanAcc');
save(strcat(recordingFolder, 'valAcc.mat'), 'valAcc');
save(strcat(recordingFolder, 'predictions.mat'), 'predictions');
save(strcat(recordingFolder, 'targets.mat'), 'targets');

% [data, targets, numFolds] = Models.LoadConvertMultipleRecordings('recordingFolder\100\', {'03-May-2022 12-08-47', '03-May-2022 12-16-04', '03-May-2022 12-20-48'});
% [meanAcc, valAcc, predictions, targets] = Models.TrainGenericModel('SVM', data, targets, numFolds)

%% Close all

close all hidden


end

