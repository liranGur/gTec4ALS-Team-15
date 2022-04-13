function [EEG] = Preprocessing(EEG, Hz, highLim, lowLim, down_srate, ...,
                               numTriggersInTrial, windowTime, numChannles, ...
                               timeBeforeTriggerToTake, triggersTime)
% Preprocessing - all the preprocessing done on recorded data
%
%INPUT:
%   EEG - raw EEG recorded. shape: (#trials, #channels, size of recording)
%   Hz - 
%   highLim - frequency high limit for low pass filter
%   lowLim - frequency low limit for high pass filter
%   down_srate - down sampling rate
%   numTriggersInTrial - 
%   windowTime - how long each trigger window should be in seconds;
%   numChannles - EEG numChannels
%   timeBeforeTriggerToTake - time of window to take befor trigger in seconds
%   triggersTime - time each trigger was activated and the end of recording time
% 
%OUTPUT:
%   EEG - processed EGG. shape: ???? #trials, #triggers, #channels, #down_srate*(windowTime+timeBeforeTriggerToTake)????

    
    EEG = split_trials(EEG, numTriggersInTrial, Hz, windowTime, numChannles, ...
                       timeBeforeTriggerToTake, triggersTime);
    EEG = pop_eegfiltnew(EEG, 'hicutoff',highLim,'plotfreqz',1); % low pass
    EEG = pop_eegfiltnew(EEG, 'locutoff',lowLim,'plotfreqz',1);  % high pass
    % downsample
    if Hz > down_srate
        EEG = pop_resample(EEG, down_srate);
    end
    
%     %Zero-phase digital filtering
%     EEG = filtfilt(EEG,b,a); %ask Ophir
%     EEG = eeg_checkset(EEG);
end
    %Median Filtering
    %Facet Method
function [res] = split_trials(EEG, triggersInTrial, Hz, windowTime, numChannles, timeBeforeTriggerToTake, ...
                              triggersTime)
% Split all the trials to triggers
% 
% INPUT:
%   EEG - full EEG recording with shape: ????
%   triggersInTrial - number of triggers in trial
%   Hz - recording frequency
%   windowTime - trigger eeg recording window time in seconds
%   numChannles - number of channels in EEG
%   timeBeforeTriggerToTake - time in seconds before trigger starts to include in trigger recording
%   triggersTime - time each trigger was activated and the end of recording time
% 
% OUTPUT:
%   res - the EEG split into triggers, shape of:???? (#trials, #channles, #triggers(trialLength), size of trigger recording)
% 
    
    first_sample_time = 

    
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
    
   