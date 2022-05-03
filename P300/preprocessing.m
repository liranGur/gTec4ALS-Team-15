function [EEG, meanTrigs] = Preprocessing(EEG, triggersTimes, trainingVec)
% Preprocessing - all the preprocessing done on recorded data
%
%INPUT:
%   EEG - raw EEG recorded. shape: (#trials, #channels, size of recording)
%   triggersTime - time each trigger was activated and the end of recording time
%   trainingVector - Triggers during training. shape: (# trials, #triggers_in_trial)
% 
%OUTPUT:
%   EEG - processed EGG. shape: #trials, #triggers_in_trial, #eeg_channels, size down sampled trigger size 

    
    
    % This code needs to be fixed to new EEG shape
    EEG = pop_eegfiltnew(EEG, 'hicutoff', Utils.Config.highLim,'plotfreqz',1); % low pass
    EEG = pop_eegfiltnew(EEG, 'locutoff',Utils.Config.lowLim,'plotfreqz',1);  % high pass

    % Spliting the trials must be the first thing to happen to allow for correct splitting because splitting is based on time stamps
    EEG1 = splitTrials(EEG, triggersTimes);
    [numTrials, ~, eegChannels, windowSize] = size(EEG1);

    % Average trigger signals per class
    classes = unique(trainingVec);
    meanTrigs = zeros(numTrials, length(classes), eegChannels, windowSize);
    for currTrial=1:numTrials    
        for class = classes.'
            meanTrigs(currTrial,class,:,:) = mean(EEG1(currTrial,trainingVec(currTrial,:) == class,:,:),2);
        end
    end
    
    % downsample
%     if Hz > Utils.Config.down_srate
%         EEG = pop_resample(EEG, Utils.Config.down_srate);
%     end
    
%     %Zero-phase digital filtering
%     EEG = filtfilt(EEG,b,a); %ask Ophir
%     EEG = eeg_checkset(EEG);
end
    %Median Filtering
    %Facet Method


   