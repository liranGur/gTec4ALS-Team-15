function [splitEEG, badTriggers] = splitTrials(EEG, triggersTimes, preTriggerRecTime, triggerWindowTime)
% Split all the trials to triggers
% 
% INPUT:
%   EEG - full EEG recording with shape: ????
%   triggersTimes - time each trigger was activated and the trial recording end time
%   preTriggerRecTime - time to keep of recording before trigger is activated (negative value means tha window will start after trigger)
%   triggerWindowTime - time to keep of recording after trigger is activated 
% 
% OUTPUT:
%   res - the EEG split into triggers, shape: #trials, #triggers_in_trial, #eeg_channels, size of trigger window)
% 

    % Extract recording info & constants
    Hz = Utils.Config.Hz;
    [numTrials, eegChannels, totalNumSamples] = size(EEG);
    totalRecordingTime = (totalNumSamples / Hz);
    numTriggersInTrial = size(triggersTimes, 2) - 1;
    preTriggerWindowSize = round(preTriggerRecTime * Hz);
    postTriggerWindowSize = round(triggerWindowTime * Hz);
    windowSize = preTriggerWindowSize + postTriggerWindowSize ;
    windowTimeDiffFromStart = -1 * preTriggerRecTime;
    
    badTriggers = {};
    badIdx = 1;
    % split to triggers
    splitEEG = zeros(numTrials, numTriggersInTrial, eegChannels, windowSize);
    for currTrial=1:numTrials
        currTrialEndTime = triggersTimes(currTrial, end);
        firstSampleRealTime = currTrialEndTime - totalRecordingTime;
        for currTrigger=1:numTriggersInTrial
            currTriggerRealTime = triggersTimes(currTrial, currTrigger);
            triggerTimeDiffFromStart = currTriggerRealTime - firstSampleRealTime;
            triggerStartSplitTime = triggerTimeDiffFromStart  + windowTimeDiffFromStart;
            currTriggerStartSampleIdx = round(triggerStartSplitTime*Hz);
            windowStartSampleIdx = currTriggerStartSampleIdx - preTriggerWindowSize;
            if windowStartSampleIdx < 1
                warning('The window start idx is less than 1 - probably recording buffer size is too small')
                windowStartSampleIdx = 1;
                badTriggers{badIdx} = struct('trial', currTrial, 'trigger', ...
                    currTrigger, 'start',windowStartSampleIdx);
                badIdx = badIdx +  1;
            end
            
            windowEndSampleIdx = currTriggerStartSampleIdx + postTriggerWindowSize - 1;            
            if windowEndSampleIdx > size(EEG,3)
                warning('The end sample index is bigger than the EEG buffer size - probably need to add more time before dumping buffer')
                windowEndSampleIdx = size(EEG,3);
                badTriggers{badIdx} = struct('trial', currTrial, 'trigger', ...
                    currTrigger, 'end',windowEndSampleIdx);
                badIdx = badIdx +  1;
            end
            
            split = squeeze(EEG(currTrial, :, windowStartSampleIdx: windowEndSampleIdx));
            split_size = size(split);
            % Pad results in case one of the sample indices is out of
            % bounds it will cause the windo to be too small. This padding
            % is required to avoid crashes
            if split_size(2) < windowSize
                final = zeros(eegChannels, windowSize);
                final(:,1:split_size(2)) = split;
                split = final;
            end
            splitEEG(currTrial, currTrigger, :, :) = split;
        end
    end
end
    