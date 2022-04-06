function [EEG] = preprocessing(EEG, Hz, highLim, lowLim, down_srate, ...,
                               triggersInTrial, windowTime, numChannles, timeBeforeTriggerToTake, timeBetweenTriggers)   
    
    EEG = split_trials(EEG, triggersInTrial, Hz, windowTime, numChannles, ...
                       timeBeforeTriggerToTake, timeBetweenTriggers);
    EEG = pop_eegfiltnew(EEG, 'hicutoff',highLim,'plotfreqz',1); % low pass
    EEG = pop_eegfiltnew(EEG, 'locutoff',lowLim,'plotfreqz',1);  % high pass
    % downsample
    if Hz > down_srate
        EEG = pop_resample(EEG, down_srate);
    end
    
%     %Zero-phase digital filtering
%     EEG = filtfilt(EEG,b,a); %ask Ophir
%     EEG = eeg_checkset(EEG);

    EEG = down_sample(EEG, down_srate, Hz);
end
    %Median Filtering
    %Facet Method
function [res] = split_trials(EEG, triggersInTrial, Hz, windowTime, numChannles, timeBeforeTriggerToTake, ...
                              timeBetweenTriggers)
% Split all the trials to triggers
% 
% INPUT:
%   EEG - full EEG recording with shape: (#trail, #channles, trail recording size)
%   triggersInTrial - number of triggers in trial
%   Hz - recording frequency
%   windowTime - trigger eeg recording window time in seconds
%   numChannles - number of channels in EEG
%   timeBeforeTriggerToTake - time in seconds before trigger starts to include in trigger recording
%   timeBetweenTriggers - time in seconds between triggers during recording
% 
% OUTPUT:
%   res - the EEG split into triggers, shape of: (#trials, #channles, #triggers(trialLength), size of trigger recording)
% 
    numOfTrials = size(EEG);
    numOfTrials = numOfTrials(1);
    preTrigRec = Hz*timeBeforeTriggerToTake;
    windowSize = Hz*windowTime + preTrigRec; 
    res = zeros(numOfTrials, numChannles, triggersInTrial, windowSize);
    for trial=1:numOfTrials
        res(trial,:,1) = EEG(trial,:,1:windowSize);
        for trigger=2:triggersInTrial
            startPos = trigger*timeBetweenTriggers*Hz - preTrigRec;
            endPos = (trigger+1)*timeBetweenTriggers*H;
            res(trial,:,trigger) = EEG(trial,:,startPos:endPos);
        end
    end
    
end
    
   