function [splitEeg, badTriggers] = splitTrials(EEG, triggersTimes)
% Split all the trials to triggers, size of triggers windows is set according to Utils.Config
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
    numTriggersInTrial = size(triggersTimes, 2) - 1;
    preTriggerWindowSize = round(Utils.Config.preTriggerRecTime * Hz);
    postTriggerWindowSize = round(Utils.Config.triggerWindowTime * Hz);
    windowSize = preTriggerWindowSize + postTriggerWindowSize ;
    triggerMoveTime = -1 * Utils.Config.preTriggerRecTime;
    
    
    badIdx = 1;
    % split to triggers
    splitEeg = zeros(numTrials, numTriggersInTrial, eegChannels, windowSize);
    for currTrial=1:numTrials
        currTrialEndTime = triggersTimes(currTrial, end);
        firstSampleRealTime = currTrialEndTime - totalRecordingTime;
        for currTrigger=1:numTriggersInTrial
            currTriggerRealTime = triggersTimes(currTrial, currTrigger);
            triggerTimeFromStart = currTriggerRealTime - firstSampleRealTime;
            triggerTimeFromStart = triggerTimeFromStart  + triggerMoveTime;
            currTriggerStartSampleIdx = round(triggerTimeFromStart*Hz);
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
            splitEeg(currTrial, currTrigger, :, :) = split;
        end
    end
end
    