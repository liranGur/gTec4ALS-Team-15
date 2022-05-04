function [splitEEG, meanTrigs, processedEEG] = Preprocessing(splitEEG, triggersTime, trainingVector)
% Preprocessing - all the preprocessing done on recorded data
%
%INPUT:
%   EEG - raw EEG recorded. shape: (#trials, #channels, size of recording)
%   triggersTime - time each trigger was activated and the end of recording time
%   trainingVector - Triggers during training. shape: (# trials, #triggers_in_trial)
% 
%OUTPUT:
%   splitEEG - Splitted EEG, Shape: num of trials, num of triggers, eggChannels, sample size
%   meanTrigs - EEG after split and mean
%               Shape: #trials, #classes, eegChannels, window size
%   processedEEG - EEG after all preprocessing: low&high pass, split to triggres, mean triggres, downsmapling
%                  shape: #trials, #classes, eegChannels, downsampled window size

    
    % This code needs to be fixed to new EEG shape
%     EEG = pop_eegfiltnew(EEG, 'hicutoff', Utils.Config.highLim,'plotfreqz',1); % low pass
%     EEG = pop_eegfiltnew(EEG, 'locutoff',Utils.Config.lowLim,'plotfreqz',1);  % high pass


    splitEEG = splitTrials(splitEEG, triggersTime);
    [numTrials, ~, eegChannels, windowSize] = size(splitEEG);
    
    % Average trigger signals per class
    classes = unique(trainingVector);
    meanTrigs = zeros(numTrials, length(classes), eegChannels, windowSize);
    for currTrial=1:numTrials    
        for class = classes.'
            meanTrigs(currTrial,class,:,:) = mean(splitEEG(currTrial,trainingVector(currTrial,:) == class,:,:),2);
        end
    end
    
    downSampledWindowSize = round(windowSize*(Utils.Config.downSampleRate/Utils.Config.Hz));
    processedEEG = zeros(numTrials, length(classes), eegChannels, downSampledWindowSize);
  
    for i =1:length(meanTrigs)
        for j=1:length(meanTrigs(i))
            %bandpass
            EEG_pass = squeeze(meanTrigs(i,j,:,:));
            %resampling
            if Utils.Config.Hz > Utils.Config.downSampleRate
                EEG_pass_trans = resample(EEG_pass.', Utils.Config.downSampleRate, ...
                    Utils.Config.Hz);
                EEG_pass = EEG_pass_trans.';
            end
            processedEEG(i,j,:,:) = EEG_pass;
        end
    end   
end

% Tests
%             EEG_mirror = [squeeze(meanTrigs(i,j,:,:)) squeeze(flip(meanTrigs(i,j,:,:),2))];
% d = designfilt('bandpassiir', ...       % Response type
%                'StopbandFrequency1',0.5, ...    % Frequency constraints
%                'PassbandFrequency1',0.6, ...
%                'PassbandFrequency2',1, ...
%                'StopbandFrequency2',1.1, ...
%                'StopbandAttenuation1',40, ...   % Magnitude constraints
%                'PassbandRipple',1, ...
%                'StopbandAttenuation2',50, ...
%                'DesignMethod','ellip', ...      % Design method
%                'MatchExactly','passband', ...   % Design method options
%                'SampleRate',Utils.Config.Hz);               % Sample rate
%             
%             
% x = wform' + 0.25*randn(500,1);
% x(x < 0) = 0;
% y = filtfilt(d,x);
% y1 = filter(d,x);
% y1 = 0;
% axis([0 500 -1.25 1.25])
% subplot(2,1,1)
% plot([y y1])
% title('Filtered Waveforms')
% legend('Zero-phase Filtering','Conventional Filtering')
% 
% subplot(2,1,2)
% plot(x)
% title('Original Waveform')
%             EEG_tran = bandpass(EEG_mirror.', [0.5 40], Utils.Config.Hz);
%             EEG_pass = EEG_tran.';
%             EEG_pass = EEG_pass(:,size(EEG_pass,2)/2);
   