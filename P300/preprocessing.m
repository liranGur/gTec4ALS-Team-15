function [EEG] = preprocessing(EEG, Hz, highLim, lowLim, down_srate)
    srate = Hz; 
    
    % low-pass filter
    EEG = pop_eegfiltnew(EEG, 'hicutoff',highLim,'plotfreqz',1);    % removes data above
    EEG = eeg_checkset(EEG);

    % High-pass filter
    EEG = pop_eegfiltnew(EEG, 'locutoff',lowLim,'plotfreqz',1);     % removes data under
    EEG = eeg_checkset(EEG);
    
%     %Zero-phase digital filtering
%     EEG = filtfilt(EEG,b,a); %ask Ophir
%     EEG = eeg_checkset(EEG);
    
    %downsample data
    if srate > down_srate
        EEG = pop_resample(EEG, down_srate);
    end
    EEG = eeg_checkset(EEG);
end
    %Median Filtering
    %Facet Method
    
   