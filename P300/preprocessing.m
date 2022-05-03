function [splitEEG, meanTrigs, splitDownSampledEeg] = Preprocessing(splitEEG, triggersTime, trainingVector)
% Preprocessing - all the preprocessing done on recorded data
%
%INPUT:
%   EEG - raw EEG recorded. shape: (#trials, #channels, size of recording)
%   triggersTime - time each trigger was activated and the end of recording time
%   trainingVector - Triggers during training. shape: (# trials, #triggers_in_trial)
% 
%OUTPUT:
%   splitEEG - Splitted EEG, Shape: num of trials, num of triggers, eggChannels, sample size
%   meanTrigs - EEG after preprocessing: low&high pass, split to triggres, mean triggres, downsmapling
%               shape: #trials, #classes, eegChannels, downsampled sample size
%   splitDownSampledEeg - splitted pre processed downsampled but not mean EEG
%                         shape: #trials, #triggers, eggChannels, downsampled sample size

    
    % This code needs to be fixed to new EEG shape
%     EEG = pop_eegfiltnew(EEG, 'hicutoff', Utils.Config.highLim,'plotfreqz',1); % low pass
%     EEG = pop_eegfiltnew(EEG, 'locutoff',Utils.Config.lowLim,'plotfreqz',1);  % high pass


    splitEEG = splitTrials(splitEEG, triggersTime);
    [numTrials, ~, eegChannels, windowSize] = size(splitEEG);
    
    splitDownSampledEeg = splitEEG;
    % downsample
%     if Hz > Utils.Config.down_srate
%         EEG = pop_resample(EEG, Utils.Config.down_srate);
%     end

    % Average trigger signals per class
    classes = unique(trainingVector);
    meanTrigs = zeros(numTrials, length(classes), eegChannels, windowSize);
    for currTrial=1:numTrials    
        for class = classes.'
            meanTrigs(currTrial,class,:,:) = mean(splitEEG(currTrial,trainingVector(currTrial,:) == class,:,:),2);
        end
    end
    
%     %Zero-phase digital filtering
%     EEG = filtfilt(EEG,b,a); %ask Ophir
%     EEG = eeg_checkset(EEG);
end
    %Median Filtering
    %Facet Method


   