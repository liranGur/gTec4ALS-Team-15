function [] = OfflinePipline()
% This functions starts the P300 offline training pipeline

baseFolder = uigetdir('C:/Subjects/', ...
    'Choose Desired Directory for Saving Recordings');

%% Recording Parameter Setting
[Hz, triggersInTrial, numClasses, subId, numTrials, timeBetweenTriggers, oddBallProb, ...
    calibrationTime, pauseBetweenTrials, triggerBankFolder] = GUIFiles.ParametersGui();

startingNormalTriggers = 3;
eegChannels = 16;
recordingFolder = [baseFolder int2str(subId)];

%% Training
[EEG, fullTrainingVec, expectedClasses] = ...
    OfflineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials, numTrials, ...
                    numClasses, oddBallProb, triggersInTrial, startingNormalTriggers, ...
                    Hz, eegChannels, triggerBankFolder, 1);

save(strcat(recordingFolder, 'trainingSequences.mat'), 'fullTrainingVec');
save(strcat(recordingFolder, 'EEG.mat'), 'EEG');
save(strcat(recordingFolder, 'trainingLabels.mat'), 'expectedClasses');
parametersToSave = struct('timeBetweenTriggers', timeBetweenTriggers, ...
                           'calibrationTime', calibrationTime, ...
                           'pauseBetweenTrials',pauseBetweenTrials, ...
                           'numTrials', numTrials, ... 
                           'startingNormalTriggers', startingNormalTriggers, ...
                           'numClasses', numClasses, ...
                           'oddBallProb', oddBallProb, ...
                           'triggersInTrial', triggersInTrial, ...
                           'Hz', Hz);
save(strcat(recordingFolder, 'parameters.mat'), 'parametersToSave')


%% PreProcessing

highLim = 100;
lowLim = 0;
downSampleRate = 50;
triggerWindowTime = 1;
preTriggerRecTime = 0.2;

processedEEG = preprocessing(EEG, Hz, highLim, lowLim, downSampleRate, triggersInTrial, triggerWindowTime, ...
                             numChannles, preTriggerRecTime, timeBetweenTriggers);

save(strcat(recordingFolder, 'processedEEG.mat'), 'processedEEG');
end

