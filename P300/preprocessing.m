function [splitEEG, meanTrigs, processedEEG] = preprocessing(EEG, triggersTimes, trainingVector, preTriggerRecTime, triggerWindowTime)
% Preprocessing - all the preprocessing done on recorded data
%
%INPUT:
%   EEG - raw EEG recorded. shape: (#trials, #channels, size of recording)
%   triggersTime - time each trigger was activated and the end of recording time
%   trainingVector - Triggers during training. shape: (# trials, #triggers_in_trial)
%   preTriggerRecTime - time to keep of recording before trigger is activated (negative value means tha window will start after trigger)
%   triggerWindowTime - time to keep of recording after trigger is activated 
% 
%OUTPUT:
%   splitEEG - Splitted EEG, Shape: num of trials, num of triggers, eggChannels, sample size
%   meanTrigs - EEG after split and mean
%               Shape: #trials, #classes, eegChannels, window size
%   processedEEG - EEG after all preprocessing: low&high pass, split to triggres, mean triggres, downsmapling
%                  shape: #trials, #classes, eegChannels, downsampled window size

%% Bandpass TODO
    bandpassedEEG = EEG;

%% Splitting
    [splitEEG, ~] = splitTrials(bandpassedEEG, triggersTimes, preTriggerRecTime, triggerWindowTime);
    [numTrials, ~, eegChannels, windowSize] = size(splitEEG);
    classes = unique(trainingVector);

%% Downsample    
    meanTrigs = averageTriggersByClass(splitEEG, numTrials, classes, eegChannels, windowSize, trainingVector);

%% DownSampling
    processedEEG = downsampleEEG(meanTrigs, numTrials, classes, eegChannels, windowSize);

end

function [meanTrigs] = averageTriggersByClass(splitEEG, numTrials, classes, eegChannels, windowSize, trainingVector)
% averageTriggersByClass - average all triggers of same class in each trial
% 
% INPUTS:
%   - splitEEG - EEG data with shape: #trials, #triggers, #channels, sample size

    meanTrigs = zeros(numTrials, length(classes), eegChannels, windowSize);
    
    for currTrial=1:numTrials    
        for class = classes.'
            meanTrigs(currTrial,class,:,:) = mean(splitEEG(currTrial,trainingVector(currTrial,:) == class,:,:),2);
        end
    end
end

function [processedEEG] = downsampleEEG(splitEEg, numTrials, classes, eegChannels, windowSize)
% downSampleEEG - downsamples EEG
% 
% INPUTS:
%   splitEEG - EEG data with shape: #trials, #triggers(can be mean triggers or anything else), #channels, sample size

    downSampledWindowSize = ceil(windowSize*(Utils.Config.downSampleRate/Utils.Config.Hz));
    processedEEG = zeros(numTrials, length(classes), eegChannels, downSampledWindowSize);
  
    for i =1:size(splitEEg, 1)
        for j=1:size(splitEEg, 2)
            squeezedEEG = squeeze(splitEEg(i,j,:,:));
            if Utils.Config.Hz > Utils.Config.downSampleRate
                EEG_pass_trans = resample(squeezedEEG.', Utils.Config.downSampleRate, ...
                    Utils.Config.Hz);
                squeezedEEG = EEG_pass_trans.';
            end
            processedEEG(i,j,:,:) = squeezedEEG;
        end
    end 
end