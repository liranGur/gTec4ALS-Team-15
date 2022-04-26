function [] = OfflinePipline()
% OfflinePipline - This functions starts the P300 offline training pipeline
% recording, preprocessing and creating a model

%clear any previous open windows - hopefully.
close all; clear; clc;

baseFolder = uigetdir('C:/P300Recordings/', ...
    'Choose Desired Directory for Saving Recordings');

%% Recording Parameter Setting
[is_visual, triggersInTrial, numClasses, subId, numTrials, timeBetweenTriggers, oddBallProb, ...
    calibrationTime, pauseBetweenTrials, triggerBankFolder, beforeJitterTime] = GUIFiles.ParametersGui();


recordingFolder = [baseFolder '\' int2str(subId) '\' strrep(datestr(now), ':','-') '\'];
mkdir(recordingFolder);

%% Training
[EEG, trainingVec, expectedClasses, triggersTimes] = ...
    OfflineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials, numTrials, beforeJitterTime, ...
                    numClasses, oddBallProb, triggersInTrial, ...
                    triggerBankFolder, is_visual);

% save(strcat(recordingFolder, 'trainingSequences.mat'), 'trainingVec');
% save(strcat(recordingFolder, 'EEG.mat'), 'EEG');
% save(strcat(recordingFolder, 'trainingLabels.mat'), 'expectedClasses');
% save(strcat(recordingFolder, 'triggersTime.mat'), 'triggersTimes');
% 
% parametersToSave = struct('timeBetweenTriggers', timeBetweenTriggers, ...
%                            'calibrationTime', calibrationTime, ...
%                            'pauseBetweenTrials',pauseBetweenTrials, ...
%                            'numTrials', numTrials, ...
%                            'beforeJitterTime', beforeJitterTime, ...
%                            'startingNormalTriggers', Utils.Config.startingNormalTriggers, ...
%                            'numClasses', numClasses, ...
%                            'oddBallProb', oddBallProb, ...
%                            'triggersInTrial', triggersInTrial, ...
%                            'triggerBankFolder', triggerBankFolder, ...
%                            'mode', is_visual, ...
%                            'Hz', Utils.Config.Hz);
% save(strcat(recordingFolder, 'parameters.mat'), 'parametersToSave')


%% PreProcessing

processedEEG = Preprocessing(EEG, triggersTimes, trainingVec);
% 
save(strcat(recordingFolder, 'processedEEG.mat'), 'processedEEG');

%% Close all

close all hidden


end

