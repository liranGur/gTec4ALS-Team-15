function [] = OfflinePipline()
% This functions starts the P300 offline training pipeline

%% Create Simulink Object
USBobj          = 'USBamp_offline';
AMPobj          = [USBobj '/g.USBamp UB-2016.03.01'];
IMPobj          = [USBobj '/Impedance Check'];
SampleSizeObj   = [USBobj '/Sample Size'];
scopeObj        = [USBobj '/g.SCOPE'];          % amsel TODO WHAT Is THIS

% RestDelayobj    = [USBobj '/Resting Delay'];
% ChunkDelayobj   = [USBobj '/Chunk Delay'];

%% Parameter Setting - until we have GUI for this
baseStartLen = 3;
sequenceLength = 20;
timeBetweenTriggers = 1;        % in seconds
Hz = 512;
eegChannels = 16;
recordingFolder = 'tmp';
calibrationTime = 20;           % in seconds
pauseBetweenTrails = 10;        % in seconds
numTrails = 3;
numClasses = 3;
oddBallProb = 0.2;



%% Start training
[EEG, fullTrainingVec, expectedClasses] = ...
    OfflineTraining(timeBetweenTriggers, calibrationTime,pauseBetweenTrails, numTrails, ...
                    numClasses, oddBallProb, sequenceLength, baseStartLen, recordingFolder, ...,
                    USBobj, Hz);

save(strcat(recordingFolder, 'trainingSequences.mat'), 'fullTrainingVec');
save(strcat(recordingFolder, 'EEG.mat'), 'EEG');
save(strcat(recordingFolder, 'trainingLabels.mat'), 'expectedClasses');
parametersToSave = struct('timeBetweenTriggers', timeBetweenTriggers, ...
                           'calibrationTime', calibrationTime, ...
                           'pauseBetweenTrails',pauseBetweenTrails, ...
                           'numTrails', numTrails, ... 
                           'startingNormalTriggers', startingNormalTriggers, ...
                           'numClasses', numClasses, ...
                           'oddBallProb', oddBallProb, ...
                           'sequenceLength', sequenceLength, ...
                           'baseStartLen', baseStartLen, ...
                           'Hz', Hz, ...
                           'trailTime', trailTime);
save(strcat(recordingFolder, 'parameters.mat'), 'parametersToSave')

% PreProcess & modelsing??
end

