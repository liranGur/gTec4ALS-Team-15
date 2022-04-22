function [EEG] = Preprocessing(EEG, triggersTime, trainingVector)
% Preprocessing - all the preprocessing done on recorded data
%
%INPUT:
%   EEG - raw EEG recorded. shape: (#trials, #channels, size of recording)
%   triggersTime - time each trigger was activated and the end of recording time
%   trainingVector - Triggers during training. shape: (# trials, #triggers_in_trial)
% 
%OUTPUT:
%   EEG - processed EGG. shape: #trials, #triggers_in_trial, #eeg_channels, size down sampled trigger size 

    % Spliting the trials must be the first thing to happen to allow for correct splitting because splitting is based on time stamps
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
    
   