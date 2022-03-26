function [] = OfflinePipline()
% This functions starts the P300 offline training pipeline

baseFolder = uigetdir('C:/Subjects/', ...
    'Choose Desired Directory for Saving Recordings');

%% Create Simulink Object
USBobj          = 'USBamp_offline';
IMPobj          = [USBobj '/Impedance Check'];


%% Parameter Setting

[Hz, trialLength, numClasses, subId, numTrials, timeBetweenTriggers, oddBallProb, ...
    calibrationTime, pauseBetweenTrials] = GUIFiles.ParametersGui(IMPobj);

startingNormalTriggers = 3;
eegChannels = 16;
recordingFolder = [baseFolder int2str(subId)];


%% Training
[EEG, fullTrainingVec, expectedClasses] = ...
    OfflineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials, numTrials, ...
                    numClasses, oddBallProb, trialLength, startingNormalTriggers, ...
                    USBobj, Hz, eegChannels)

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
                           'trialLength', trialLength, ...
                           'Hz', Hz);
save(strcat(recordingFolder, 'parameters.mat'), 'parametersToSave')

end

