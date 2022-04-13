function [EEG, fullTrainingVec, expectedClasses, triggersTime] = ... 
    OfflineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials, numTrials, timeBeforeJitter,...
                    numClasses, oddBallProb, triggersInTrial, ...
                    triggerBankFolder, is_visual)
% OfflineTraining - This function is responsible for offline training and
% recording EEG data
% INPUT:
%   - timeBetweenTriggers - in seconds
%   - calibrationTime - system calibration time in seconds
%   - pauseBetweenTrials - pause time in seconds
%   - numTrials
%   - timeBeforeJitter - time in seconds before mark jitter happens (jitter blank screen or removing the selected trigger) This is relevant only for visual
%   - numClasses - odd ball classe (e.g., 1-> only one odd ball and baseline)
%   - oddBallProb - in [0,1]
%   - triggersInTrial - number of triggers in a trial
%   - triggerBankFolder - relative/absolute path to selected trigger bank (folder with images/audio for training)
%   - is_visual - visual or auditory P300

% OUTPUT:
%   EEG - EEG signal of training. shape: (# trials, # EEG channels, trial sample size) 
%   fullTrainingVec - Triggers during training. shape: (# trials, trial length)
%   expectedClasses - class number the subject neeeds to focus on in each trial
%   triggersTime - system time of each trigger showing and the buffer dump time
%


%% Set up parameters
trialTime = triggersInTrial*timeBetweenTriggers + Utils.Config.pretrialSafetyBuffer;
eegSampleSize = Utils.Config.Hz*trialTime; 
% recordingBuffer = setUpRecordingSimulink(Utils.Config.Hz, eegSampleSize);

%% Load Train Samples
[trainingSamples, diffTrigger, classNames] = loadTrainingSamples(triggerBankFolder, is_visual);

%% Setup & Callibrate System

[fig, ax] = Utils.DisplaySetUp();

% Show a message that declares that training is about to begin
displayText(['System is calibrating.' sprintf('\n') ...
             'The training session will begin shortly.'])
pause(calibrationTime)

%% Training Setup
if is_visual
    activateTrigger = @(images, idx) activateVisualTrigger(images, idx, diffTrigger, timeBeforeJitter);
else
    activateTrigger = @(sounds, idx) activateAudioTrigger(sounds, idx);
end

fullTrainingVec = ones(numTrials, triggersInTrial);
expectedClasses = zeros(numTrials, 1);
EEG = zeros(numTrials, Utils.Config.eegChannels, eegSampleSize);
triggersTime = zeros(numTrials, (triggersInTrial+1));

%% Training

for currTrial = 1:numTrials
    % Prepare Trial
    trainingVec = Utils.TrainingVecCreator(numClasses, oddBallProb, triggersInTrial, is_visual);
    assert(all(trainingVec <= (numClasses + 1)), 'Sanity check training Vector')
    targetClass = round((numClasses-1)*rand) + 2;
    expectedClasses(currTrial) = targetClass;
    assert(targetClass > 1 & targetClass <= (numClasses+1), 'Sanity check target class')
    fullTrainingVec(currTrial, : ) = trainingVec;
    
    displayText(['Starting Trial ' int2str(currTrial) sprintf('\n') ...
                 ' Please count the apperances of class'  sprintf('\n')...
                 classNames(targetClass)]);
    
    % Show base image for a few seconds before start
    if is_visual
        pause(Utils.Config.preTrialPause);
        dispalyImageWrapper(diffTrigger)
    end
    
    % Short wait before trial starts
    pause(Utils.Config.preTrialPause);
    
    % Trial - play triggers
    for currTrigger=1:triggersInTrial 
        currClass = trainingVec(currTrigger);
        currTriggerTime = activateTrigger(trainingSamples, currClass);
        triggersTime(currTrial, currTrigger+1) = currTriggerTime;
        pause(timeBetweenTriggers + rand*Utils.Config.maxRandomTimeBetweenTriggers)  % use random time diff between triggers
    end
    
    %     EEG(currTrial, :, :) = recordingBuffer.OutputPort(1).Data'; 
    % get EEG dump time of trial
    triggersTime(currTrial,(triggersInTrial+1)) = now;    
    
    % End of Trial
    set(fig, 'color', 'black');          % imshow removes background color, therefore we need to set it again before showing more text
    displayText(['Finished Trial ' int2str(currTrial) sprintf('\n') ...
                 'Pausing for: ' int2str(pauseBetweenTrials) ' seconds before next trial.']);

    if ~is_visual % Play end sound if needed
        sound(diffTrigger, getSoundFs());
    end

    pause(pauseBetweenTrials)
end


end


%% Loading triggers functions

function [trainingSamples, diffTrigger, classNames] = loadTrainingSamples(triggerBankFolder, is_visual)
% loadTrainingSamples - load images / audio for training
%
% OUTPUT:
%   trainingSamples - struct of all the triggers loaded according to their class index
%   diffTrigger - This is an image/sound that isn't part of the triggers but is needed for training.
%       In visual mode this is the basic shapes without selecting any (yellow rectangle)
%       In auditory mode this is the end trial sound
%   classNames - struct of each trigger name


    % This function does the actual loading from flie
    function [trigger] = load_trigger_func(varargin)
       path =  varargin{1,end};
       path = path{1};
       if is_visual
           trigger = imread(path);
       else
           trigger = audioread(path);
       end
    end


    files = dir(triggerBankFolder);
    for i = 1:length(files)
        file_names{i} = files(i).name;
    end
    file_names = sort(file_names);
    file_names = file_names(3:length(file_names));      % remove . & .. from file_names
    classNames{1} = 'baseline';
    
    diffTrigger = load_trigger_func(strcat(triggerBankFolder, '\', file_names(1)));
    for i=2:length(file_names)
        trainingSamples{i-1} = load_trigger_func(strcat(triggerBankFolder, '\', file_names(i)));
        classNames{i-1} = getClassNameFromFileName(file_names(i));
    end
end


function [name] = getClassNameFromFileName(file_name)
% getClassNameFromFileName - extract the class name from the file name.
% !!! All trigger file names should follow this template:trigger<idx>_<name>.<file type> !!!
    file_name = file_name{1};
    start_loc = strfind(file_name, '_');
    start_loc = start_loc(1);
    end_loc = strfind(file_name, '.');
    end_loc = end_loc(1);
    name = file_name(start_loc:(end_loc-1));
end

%% Trigger activation

function [time] = activateVisualTrigger(trainingImages, idx, jitterImage, timeBeforeJitter)
    cla
    dispalyImageWrapper(trainingImages{idx})
    time = now;
    pause(timeBeforeJitter);
    dispalyImageWrapper(jitterImage);
end

function [time] = activateAudioTrigger(trainingSounds, idx)
    sound(trainingSounds{1, idx}, getSoundFs());
    time = now;
end

function [soundFs] = getSoundFs()
    soundFs = 49920;
end

%% Diff Functions

function [recordingBuffer] = setUpRecordingSimulink(Hz, eegSampleSize)
% setUpRecordingSimulink - creates and setups the simulink object needed for training
%  
% INPUT - 
%   Hz - recording frequency
%   eegSampleSize - Size of buffer dump
%
% OUTPUT - 
%   recordingBuffer - simulink recording buffer object

    [usbObj, scopeObj, impObj, ampObj] = Utils.CreateSimulinkObj();

    % open Simulink
    open_system(['GUIFiles/' usbObj])

%     set_param(ampObj, 'Hz', num2str(Hz));           % TODO check how hz is configured in slx

    % Set simulink recording buffer size 
    SampleSizeObj = [usbObj '/Chunk Delay'];        % Todo try to change this name
    set_param(SampleSizeObj,'siz',num2str(eegSampleSize));

    Utils.startSimulation(inf, usbObj);
    open_system(scopeObj);

    recordingBuffer = get_param(SampleSizeObj,'RuntimeObject');
end


function dispalyImageWrapper(image)
    imshow(image, 'InitialMagnification','fit')
end

function displayText(textToDisplay)
    cla
    text(0.5,0.5, textToDisplay, ...
         'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
end




