function [EEG, triggersTime, fullTrainingVec] = OnlineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials,...
                                              timeBeforeJitter,...
                                              numClasses, oddBallProb, triggersInTrial, ...
                                              triggerBankFolder, is_visual)
%ONLINETRAINING Summary of this function goes here
%   Detailed explanation goes here


%% Set up recording
[eegSampleSize, recordingBuffer, trainingSamples, diffTrigger, classNames, activateTrigger, fig] = ...
    RecordingSetup(timeBetweenTriggers, calibrationTime, triggersInTrial, triggerBankFolder, timeBeforeJitter, is_visual);


fullTrainingVec = ones(numTrials, triggersInTrial);
EEG = zeros(numTrials, Utils.Config.eegChannels, eegSampleSize);
triggersTime = zeros(numTrials, (triggersInTrial+1));

%%
end

