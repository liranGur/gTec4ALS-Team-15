function [trainingVec, trialEEG, triggersTime, backupTimes] = SingleTrialRecording(...
                                                        numClasses, oddBallProb, triggersInTrial, is_visual, ...
                                                        eegSampleSize, recordingBuffer, ...
                                                        maxRandomTimeBetweenTriggers, preTrialPause, timeBetweenTriggers , ...
                                                        activateTrigger, diffTrigger, trainingSamples, ...
                                                        expectedClassName, currTrialStr ...
                                                        )
% SingleTrialRecording - record a single trial. This function is generic
% for both online and offline
% 
% INPUT:
%   - numClasses - odd ball classes (e.g., 1-> only one odd ball and baseline)
%   - oddBallProb - probability of the oddball showing, in range [0,1)
%   - triggersInTrial - number of triggers shown in the trial
%   - is_visual - 1 if the subject wants visual triggers, 0 - for auditoary triggers
%   - eegSampleSize -  size of the eeg buffer for the trial
%   - recordingBuffer - the simulink object for dumping the EEG buffer
%   - maxRandomTimeBetweenTriggers -  maximum amount of time that can be added between triggers randomly in seconds
%   - preTrialPause - Time before trial starts (in viusal this is the time the base picuture is shown)
%   - timeBetweenTriggers - time to pause between triggers (doesn't include random time)
%   - activateTrigger - function to activate trigger visual or auditory. It recives 2 prameters: the training samples struct and the index of the trigger to activate
%   - diffTrigger - the differnt triggers (not idle) that is presented to the subject. In visual this is a blank screen. In auditoary this is the end sound.
%   - trainingSamples - all the training triggers to be shown/played. This is an array with size of numClasses
%   - expectedClassName - Name of expected class to tell the subject to focus on
%   - currTrialStr - number of trial as string
% 
% OUTPUT:
%   - trainingVec -  a vector with all the labels of the classes during training. size of triggersInTrial
%   - trialEEG - raw EEG data of recording, shape: #eeg channels, eeg sample size of full trial
%   - triggersTime - times the triggers were activated with last value the dump time of buffer
%   - backupTimes - times before the triggers were activated with last value the time before dumping the buffer

%% Prepare Trial
    triggersTime = zeros(triggersInTrial + 1, 1);
    backupTimes = zeros(triggersInTrial + 1, 1);
    trialEEG = zeros(Utils.Config.eegChannels, eegSampleSize);
    
    trainingVec = Utils.TrainingVecCreator(numClasses, oddBallProb, triggersInTrial, is_visual);
    assert(all(trainingVec <= (numClasses + 1)), 'Sanity check training Vector')
    Recording.DisplayTextOnFig(['Starting Trial ' currTrialStr sprintf('\n') ...
                                     ' Please count the apperances of the class ' expectedClassName]);
     
%% Pre-trial
    % Show base image for a few seconds before trial starts
    if is_visual
        pause(preTrialPause);
        Recording.DispalyImageWrapper(diffTrigger)
    end

    pause(preTrialPause);    

%% Recording Loop

    for currTrigger=1:triggersInTrial 
        currClass = trainingVec(currTrigger);
        [pre_time, currTriggerTime] = activateTrigger(trainingSamples, currClass);
        triggersTime(currTrigger) = currTriggerTime;
        backupTimes(currTrigger) = pre_time;
        pause(timeBetweenTriggers + rand * maxRandomTimeBetweenTriggers)  % use random time diff between triggers
    end
    
%% Finish trial

    pause(Utils.Config.pauseBeforeDump);
    backupTimes(triggersInTrial+1) = posixtime(datetime('now'));
    trialEEG(currTrial, :, :) = recordingBuffer.OutputPort(1).Data';    % CHANGE FOR NO RECORDING
    triggersTime(triggersInTrial+1) = posixtime(datetime('now'));
    
end

