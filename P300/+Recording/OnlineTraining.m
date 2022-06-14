function [EEG, trainingVec, triggersTime, backupTimes, fig, classesNames] = ...
    OnlineTraining(numClasses, oddBallProb, triggersInTrial, is_visual, ...
    maxRandomTimeBetweenTriggers, preTrialPause, timeBetweenTriggers, ...
    eegSampleSize, recordingBuffer, trainingSamples, diffTrigger, classesNames,...
    activateTrigger, fig)

%OnlineTraining This is the online recording process
% 
% INPUT:
%   - timeBetweenTriggers - time to pause between triggers (doesn't include random time)
%   - calibrationTime - system calibration time in seconds
%   - timeBeforeJitter - time in seconds before mark jitter happens (jitter blank screen or removing the selected trigger) This is relevant only for visual
%   - numClasses - odd ball classes (e.g., 1-> only one odd ball and baseline)
%   - oddBallProb - probability of the oddball showing, in range [0,1)
%   - triggersInTrial - number of triggers shown in the trial
%   - triggerBankFolder - relative/absolute path to selected trigger bank (folder with images/audio for training)
%   - is_visual - visual or auditory P300
%   - preTrialPause - Time before trial starts (in viusal this is the time the base picuture is shown)
%   - maxRandomTimeBetweenTriggers -  maximum amount of time that can be added between triggers randomly in seconds
% 
% OUTPUT:
%   - EEG - raw EEG data from currnet recording. shape: 1, #eeg channels, #eeg sample size
%   - trainingVec - Triggers during training. shape: 1, # trials, trial length
%   - triggersTime - times the triggers were activated with last value the dump time of buffer
%   - backupTimes - times before the triggers were activated with last value the time before dumping the buffer
%   - fig - The figure that is used to dispaly triggers and text
%   - classesNames - struct with the name of classes of the triggers
% 


    %% Recorod Subject Selection
    triggersTime = zeros(1, triggersInTrial + 1, 1);
    backupTimes = zeros(1, triggersInTrial + 1, 1);
    EEG = zeros(1, Utils.Config.eegChannels, eegSampleSize);
    trainingVec = ones(1, triggersInTrial);
    [trainingVec(1,:), EEG(1,:,:), triggersTime(1,:,:), backupTimes(1,:,:)] = ...
        Recording.SingleTrialRecording(numClasses, oddBallProb, triggersInTrial, is_visual, ...
                                       eegSampleSize, recordingBuffer, maxRandomTimeBetweenTriggers, ...
                                       preTrialPause, timeBetweenTriggers , ...
                                       activateTrigger, diffTrigger, trainingSamples, ...
                                       'Of your answer' , ' ');

    %% Finish Recording
    endTrialTxt = '<html>Finished recording your response<br>Please wait';

    set(fig, 'color', 'black');          % imshow removes background color, therefore we need to set it again before showing more text
    Recording.DisplayTextOnFig(endTrialTxt);

    if ~is_visual % Play end sound if needed
        sound(diffTrigger, getSoundFs());
    end

end

