function [EEG, fullTrainingVec, expectedClasses] = ... 
    OfflineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials, numTrials, ...
                    numClasses, oddBallProb, triggersInTrial, baseStartLen, ...
                    Hz, eegChannels, triggerBankFolder, is_visual)
% OfflineTraining - This function is responsible for offline training and
% recording EEG data
% INPUT:
%   - timeBetweenTriggers - in seconds
%   - calibrationTime - system calibration time in seconds
%   - pauseBetweenTrials - in seconds
%   - numTrials
%   - numClasses - odd ball classe (e.g., 1-> only one odd ball and baseline)
%   - oddBallProb - in [0,1]
%   - triggersInTrial - number of triggers in a trial
%   - baseStartLen - number baseline triggers in the start of each trial
%   - USBobj - Simulink object
%   - Hz - EEG recording frequency
%   - eegChannels - number of channels recorded (i.e., number of electordes)

% OUTPUT:
%   EEG - EEG signal of training. shape: (# trials, # EEG channels, trial sample size) 
%   fullTrainingVec - Triggers during training. shape: (# trials, trial length)
%   expectedClasses - Class the subject neeeds to focus on in each trial
%


preTrialPause = 2;

pretrialSafetyBuffer = 3;                       % seconds to record before trial starts
trialTime = triggersInTrial*timeBetweenTriggers + pretrialSafetyBuffer;
eegSampleSize = Hz*trialTime; 

% recordingBuffer = setUpRecordingSimulink(Hz, eegSampleSize);

%% Load Train Samples
[trainingSamples, diffTrigger, classNames] = loadTrainingSamples(triggerBankFolder, is_visual);

%% Callibrate System

Utils.DisplaySetUp();

% Show a message that declares that training is about to begin
text(0.5,0.5 ,...
    ['System is calibrating.' sprintf('\n') 'The training session will begin shortly.'], ...
    'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
pause(calibrationTime)

% Clear axis
cla

%% Record trials
if is_visual
    activateTrigger = @activateVisualTrigger;
else
    activateTrigger = @activateAudioTrigger;
end

fullTrainingVec = ones(numTrials, triggersInTrial);
expectedClasses = zeros(numTrials, 1);
EEG = zeros(numTrials, eegChannels, eegSampleSize);

for currTrial = 1:numTrials
    % Prepare Trial
    trainingVec = Utils.TrainingVecCreator(numClasses, oddBallProb, triggersInTrial, baseStartLen);  % This also needs to be updated
    targetClass = round((numClasses-1)*rand);
    expectedClasses(currTrial) = targetClass;
    fullTrainingVec(currTrial, : ) = trainingVec;
    text(0.5,0.5 ,...
        ['Starting Trial ' int2str(currTrial) sprintf('\n') 'Please count the apperances of class' classNames(targetClass)], ...
        'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    pause(preTrialPause)
    
    % Show base image for a few seconds before start
    if is_visual
        cla
        image(flip(diffTrigger, 1), 'XData', [0.25, 0.75],...
        'YData', [0.25, 0.75 * ...
        size(diffTrigger ,1)./ size(diffTrigger,2)])
        pause(3);
    end    
    
    % Trial - play triggers
    for currTrigger=1:triggersInTrial 
        currClass = trainingVec(currTrigger);
        activateTrigger(trainingSamples, currClass)
        pause(timeBetweenTriggers)
    end
    
    % End of Trial
    cla
    text(0.5,0.5 ,...
        ['Finished Trial ' int2str(currTrial) sprintf('\n') ...
         'Pausing for: ' int2str(pauseBetweenTrials) ' seconds before next trial.'], ...
         'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);

    if ~is_visual % Play end sound if needed
        sound(diffTrigger, getSoundFs());
    end
     
    pause(0.5)  % pausing as a safety buffer for final trigger recording in EEG
    EEG(currTrial, :, :) = recordingBuffer.OutputPort(1).Data'; 

    pause(pauseBetweenTrials)
end

close(MainFig)
end

function [trainingSamples, diffTrigger, classNames] = loadTrainingSamples(triggerBankFolder, is_visual)

    function [trigger] = load_func(varargin)
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
    file_names = file_names(3:length(file_names));  % remove . & .. from names
    
    diffTrigger = load_func(strcat(triggerBankFolder, '\', file_names(1)));
    for i=2:length(file_names)
        trainingSamples{i-1} = load_func(strcat(triggerBankFolder, '\', file_names(i)));
        classNames{i-1} = getClassNameFromFileName(file_names(i));
    end
end

function [name] = getClassNameFromFileName(file_name)
    file_name = file_name{1};
    start_loc = strfind(file_name, '_');
    start_loc = start_loc(1);
    end_loc = strfind(file_name, '.');
    end_loc = end_loc(1);
    name = file_name(start_loc:(end_loc-1));
end

function [recordingBuffer] = setUpRecordingSimulink(Hz, eegSampleSize) 
    [usbObj, scopeObj, impObj, ampObj] = Utils.CreateSimulinkObj();

    % open Simulink
    open_system(['GUIFiles/' usbObj])

    set_param(ampObj, 'Hz', num2str(Hz));           % TODO check how hz is configured in slx

    % Set simulink recording buffer size 
    SampleSizeObj = [usbObj '/Chunk Delay'];        % Todo try to change this name
    set_param(SampleSizeObj,'siz',num2str(eegSampleSize));

    Utils.startSimulation(inf, usbObj);
    open_system(scopeObj);

    recordingBuffer = get_param(SampleSizeObj,'RuntimeObject');
end

function activateVisualTrigger(trainingImages, idx)
    cla
    image(flip(trainingImages{idx}, 1), 'XData', [0.25, 0.75],...
        'YData', [0.25, 0.75 * ...
        size(trainingImages{idx},1)./ size(trainingImages{idx},2)])
end

function activateAudioTrigger(trainingSounds, idx)
    sound_fs = 4096;
    sound(trainingSounds{1, idx}, getSoundFs());
end

function [soundFs] = getSoundFs()
    soundFs = 49920;
end


