function [eegSampleSize, recordingBuffer, trainingSamples, diffTrigger, classNames, activateTrigger, fig] = ...
    RecordingSetup(timeBetweenTriggers, calibrationTime, triggersInTrial, triggerBankFolder, timeBeforeJitter, is_visual)
%RecordingSetup  inital setup function for running online & offline
% training it loads all needed data, creates simulink buffer and calibarates
% the system
% 
% INPUT:
%   - timeBetweenTriggers - time to pause between triggers (doesn't include random time)
%   - calibrationTime - system calibration time in seconds
%   - triggersInTrial - number of triggers shown in the trial
%   - triggerBankFolder - relative/absolute path to selected trigger bank (folder with images/audio for training)
%   - timeBeforeJitter - time in seconds before mark jitter happens (jitter blank screen or removing the selected trigger) This is relevant only for visual
%   - is_visual - visual or auditory P300
% 
% OUTPUT:
%   - eegSampleSize -  size of the eeg buffer for the trial
%   - recordingBuffer - the simulink object for dumping the EEG buffer
%   - trainingSamples - all the training triggers to be shown/played. This is an array with size of numClasses
%   - diffTrigger - the differnt triggers (not idle) that is presented to the subject. In visual this is a blank screen. In auditoary this is the end sound.
%   - classNames - strcut with the name of the classes
%   - activateTrigger - function to activate trigger visual or auditory. It recives 2 prameters: the training samples struct and the index of the trigger to activate
%   - fig - the figure used to show triggers and text to the user
% 


%% Set up parameters
eegSampleSize = CalculateRecordingBufferSize(triggersInTrial, timeBetweenTriggers);
recordingBuffer = SetUpRecordingSimulink(eegSampleSize);       % CHANGE FOR NO RECORDING

%% Load Train Samples
[trainingSamples, diffTrigger, classNames] = Utils.LoadTrainingSamples(triggerBankFolder, is_visual);

if is_visual
    activateTrigger = @(images, idx) activateVisualTrigger(images, idx, diffTrigger, timeBeforeJitter);
else
    activateTrigger = @(sounds, idx) activateAudioTrigger(sounds, idx);
end

%% Setup & Callibrate System

[fig, ~] = Utils.DisplaySetUp();

Recording.DisplayTextOnFig(['System is calibrating.' sprintf('\n') ...
                        'The training session will begin shortly.'])
pause(calibrationTime)

end

%% Simulink Setup Functions

function [eegSampleSize] = CalculateRecordingBufferSize(triggersInTrial, timeBetweenTriggers)
%CalculateRecordingBufferSize - calculate recording buffer size for simulink
%
% INPUT:
%   - triggersInTrial - number of triggers shown to the user during the trial
%   - timeBetweenTriggers - rest time between users
% 
% OUTPUS:
%   - eegSampleSize - size of recording buffer

trialTime = triggersInTrial*(timeBetweenTriggers+Utils.Config.maxRandomTimeBetweenTriggers) ...
    + Utils.Config.pretrialSafetyBuffer + Utils.Config.pauseBeforeDump;
eegSampleSize = ceil(Utils.Config.Hz*trialTime); 

end

function [recordingBuffer] = SetUpRecordingSimulink(eegSampleSize)
% SetUpRecordingSimulink - creates and setups the simulink object needed for training
%  
% INPUT - 
%   eegSampleSize - Size of buffer dump
%
% OUTPUT - 
%   recordingBuffer - simulink recording buffer object

    [usbObj, scopeObj, ~, ~] = Utils.CreateSimulinkObj();

    % open Simulink
    open_system(['GUIFiles/' usbObj])

    % Set simulink recording buffer size 
    SampleSizeObj = [usbObj '/Chunk Delay'];        % Todo try to change this name
    set_param(SampleSizeObj,'siz',num2str(round(eegSampleSize)));

    Utils.startSimulation(inf, usbObj);
    open_system(scopeObj);

    recordingBuffer = get_param(SampleSizeObj,'RuntimeObject');
end


%% Trigger activation Functions

function [time, pre_time] = activateVisualTrigger(trainingImages, idx, jitterImage, timeBeforeJitter)
% activateVisualTrigger - shows a visual trigger
%
% INPUT:
%   trainingImages - struct of all images
%   idx - index of image to display
%   jitterImage - the image to show when the trigger ends
%   timeBeforeJitter - how long the visual trigger will be
% 
% OUTPUT:
%   time - the time in ns when the trigger appeared on screen
    
    cla
    pre_time = posixtime(datetime('now'));
    Recording.DispalyImageWrapper(trainingImages{idx})
    time = posixtime(datetime('now'));
    pause(timeBeforeJitter);
    Recording.DispalyImageWrapper(jitterImage);
end


function [time] = activateAudioTrigger(trainingSounds, idx)
% activateAudioTrigger - plays the audio trigger
% 
% INPUT:
%     trainingSounds - struct of all training sounds
%     idx - index of sound to play
% 
% OUTPUT:
%     time - the time the sound trigger was played

    sound(trainingSounds{1, idx}, getSoundFs());
    time = posixtime(datetime('now'));
end


function [soundFs] = getSoundFs()
    soundFs = 49920;
end
