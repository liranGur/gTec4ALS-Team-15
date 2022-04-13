classdef Config
    %CONFIG Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Constant)
        %% Train & Record parameters
        Hz = 512;
        startingNormalTriggers = 3;     % number baseline triggers in the start of each trial
        eegChannels = 16;
        pretrialSafetyBuffer = 3;       % Time in seconds to add to buffer for safety reasons (not to lose recording data)
        preTrialPause = 2;              % How long to wait in seconds before starting trial after showing the new trial window
        maxRandomTimeBetweenTriggers = 0.3;
        %% Preprocessing paramters
        highLim = 100;
        lowLim = 0;
        downSampleRate = 50;
        triggerWindowTime = 1;
        preTriggerRecTime = 0.2;
    end
    
    methods
    end
    
end

