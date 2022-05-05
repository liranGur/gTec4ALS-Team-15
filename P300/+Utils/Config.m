classdef Config
    %CONFIG Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Constant)
        %% Train & Record parameters
        Hz = 512;
        startingNormalTriggers = 3;          % number baseline triggers in the start of each trial
        eegChannels = 16;
        pretrialSafetyBuffer = 20;           % Time in seconds to add to buffer for safety reasons (not to lose recording data)
        preTrialPause = 4;                   % How long to wait in seconds before starting trial after showing the new trial window
        maxRandomTimeBetweenTriggers = 0.3;
        
        %% Preprocessing paramters
        highLim = 100;                      % Low pass frequency filter value
        lowLim = 0;                         % High pass frequency filter value
        downSampleRate = 40;                % Downsampling rate
        triggerWindowTime = 0.6;            % Size of each trigger EEG window (doesn't include time before trigger)
        preTriggerRecTime = -0.15;          % Time before trigger to include in trigger window. Use negative values to start splitting some time after the trigger
    end
    
    methods
    end
    
end

