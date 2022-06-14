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
        pauseBeforeDump = 0.6;               % This pause is done before dumping the simulink buffer to make sure we have the p300 of the last trigger
        chanMap = containers.Map({'PO3', 'FCZ', 'O1', 'Oz', 'C3', 'P3', 'CZ', 'PO4', 'C4', 'PZ', 'POZ', 'PO7', 'CPZ', 'PO8', 'P4', 'FZ'}, ...
                                 { 1,     2,     3,    4,   5,    6,    7,    8,     9,    10,   11,    12,    13,    14,   15,  16});

        %% Preprocessing paramters
        highLim = 100;                      % Low pass frequency filter value
        lowLim = 0;                         % High pass frequency filter value
        downSampleRate = 60;                % Downsampling rate
        triggerWindowTime = 1;              % Size of each trigger EEG window (doesn't include time before trigger)
        preTriggerRecTime = 0.2;            % Time before trigger to include in trigger window. Use negative values to start splitting some time after the trigger
        
        % for using models set the following:
%             * triggerWindowTime = 0.6
%             * preTriggerRecTime = -0.2

        %% Others
        modelDirName = 'subjectsModels'
    end
    
    methods
    end
    
end

