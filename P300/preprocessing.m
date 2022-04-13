function [EEG] = Preprocessing(EEG, triggersTime)
% Preprocessing - all the preprocessing done on recorded data
%
%INPUT:
%   EEG - raw EEG recorded. shape: (#trials, #channels, size of recording)
%   triggersTime - time each trigger was activated and the end of recording time
% 
%OUTPUT:
%   EEG - processed EGG. shape: #trials, #triggers_in_trial, #eeg_channels, size down sampled trigger size 

    EEG = splitTrials(EEG, triggersTime);
    
    % This code needs to be fixed to new EEG shape
    EEG = pop_eegfiltnew(EEG, 'hicutoff', Utils.Config.highLim,'plotfreqz',1); % low pass
    EEG = pop_eegfiltnew(EEG, 'locutoff',Utils.Config.lowLim,'plotfreqz',1);  % high pass
    % downsample
    if Hz > Utils.Config.down_srate
        EEG = pop_resample(EEG, Utils.Config.down_srate);
    end
    
%     %Zero-phase digital filtering
%     EEG = filtfilt(EEG,b,a); %ask Ophir
%     EEG = eeg_checkset(EEG);
end
    %Median Filtering
    %Facet Method

function [splitEeg] = splitTrials(EEG, triggersTimes)
% Split all the trials to triggers, size is set according to Utils.Config
% 
% INPUT:
%   EEG - full EEG recording with shape: ????
%   triggersTimes - time each trigger was activated and the trial recording end time
% 
% OUTPUT:
%   res - the EEG split into triggers, shape: #trials, #triggers_in_trial, #eeg_channels, size of trigger window)
% 

    % Extract recording info & constants
    Hz = Utils.Config.Hz;
    eegShape = size(EEG);
    totalNumSamples = eegShape(3);
    numTrials = eegShape(1);
    eegChannels = eegShape(2);
    totalRecordingTime = Hz / totalNumSamples;
    totalRecordingTime = totalRecordingTime*6e10;            % Convert to nanoseconds
    numTriggersInTrial = size(triggersTimes);
    numTriggersInTrial = numTriggersInTrial(2) - 1;
    preTriggerWindowSize = Utils.Config.preTriggerRecTime * Hz ;
    postTriggerWindowSize = Utils.Config.triggerWindowTime * Hz;
    windowSize = preTriggerWindowSize + postTriggerWindowSize ;
    
    splitEeg = zeros(numTrials, numTriggersInTrial, eegChannels, windowSize);
    
    % split to triggers
    for currTrial=1:numTrials
        currTrialEndTime = triggersTimes(currTrial, numTriggersInTrial + 1);
        currTrialStartTime = currTrialEndTime - totalRecordingTime;
        for currTrigger=1:numTriggersInTrial
            currTriggerStartSampleIdx = triggersTimes(currTrial, currTrigger) - currTrialStartTime;
            windowStartSampleIdx = currTriggerStartSampleIdx - preTriggerWindowSize;
            windowEndSampleIdx = currTriggerStartSampleIdx + postTriggerWindowSize;
            splitEeg(currTrial, currTrigger, :, :) = EEG(currTrial, :, windowStartSampleIdx: windowEndSampleIdx);
        end
    end
end
    
   