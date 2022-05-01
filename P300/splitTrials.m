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
    [numTrials, eegChannels, totalNumSamples] = size(EEG);
    totalRecordingTime = (totalNumSamples / Hz);
    numTriggersInTrial = size(triggersTimes);
    numTriggersInTrial = numTriggersInTrial(2) - 1;
    preTriggerWindowSize = round(Utils.Config.preTriggerRecTime * Hz);
    postTriggerWindowSize = round(Utils.Config.triggerWindowTime * Hz);
    windowSize = preTriggerWindowSize + postTriggerWindowSize ;
    
    % split to triggers
    splitEeg = zeros(numTrials, numTriggersInTrial, eegChannels, windowSize);
    for currTrial=1:numTrials
        currTrialEndTime = triggersTimes(currTrial, numTriggersInTrial + 1);
        currTrialStartTime = currTrialEndTime - totalRecordingTime;
        for currTrigger=1:numTriggersInTrial
            currTriggerStartSampleIdx = round(triggersTimes(currTrial, currTrigger) - currTrialStartTime);
            windowStartSampleIdx = currTriggerStartSampleIdx - preTriggerWindowSize;
            if windowStartSampleIdx < 1
                windowStartSampleIdx = 1;
            end
            windowEndSampleIdx = currTriggerStartSampleIdx + postTriggerWindowSize;
            split = squeeze(EEG(currTrial, :, windowStartSampleIdx: windowEndSampleIdx));
            split_size = size(split);
            if split_size(2) < windowSize
                final = zeros(eegChannels, windowSize);
                final(:,1:split_size(2)) = split;
                split = final;
            end
            splitEeg(currTrial, currTrigger, :, :) = split;
        end
    end
end
    