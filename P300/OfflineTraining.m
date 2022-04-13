function [EEG, fullTrainingVec, expectedClasses, triggersTime] = ... 
    OfflineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials, numTrials, ...
                    numClasses, oddBallProb, triggersInTrial, ...
                    triggerBankFolder, is_visual)
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
%   - baseStartLen - 
%   - USBobj - Simulink object

% OUTPUT:
%   EEG - EEG signal of training. shape: (# trials, # EEG channels, trial sample size) 
%   fullTrainingVec - Triggers during training. shape: (# trials, trial length)
%   expectedClasses - Class the subject neeeds to focus on in each trial
%   triggersTime - system time of each trigger showing and the buffer dump time
%


trialTime = triggersInTrial*timeBetweenTriggers + Utils.Config.pretrialSafetyBuffer;
eegSampleSize = Utils.Config.Hz*trialTime; 

% recordingBuffer = setUpRecordingSimulink(Utils.Config.Hz, eegSampleSize);

%% Load Train Samples
[trainingSamples, diffTrigger, classNames] = loadTrainingSamples(triggerBankFolder, is_visual);

%% Callibrate System

% Utils.DisplaySetUp();
figure()

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
EEG = zeros(numTrials, Utils.Config.eegChannels, eegSampleSize);
triggersTime = zeros(numTrials, (triggersInTrial+1));

for currTrial = 1:numTrials
    % Prepare Trial
    trainingVec = Utils.TrainingVecCreator(numClasses, oddBallProb, triggersInTrial, is_visual);
    assert(all(trainingVec <= (numClasses + 1)), 'Sanity check training Vector')
    targetClass = round((numClasses-1)*rand) + 2;
    expectedClasses(currTrial) = targetClass;
    assert(targetClass > 1 & targetClass <= (numClasses+1), 'Sanity check target class')
    fullTrainingVec(currTrial, : ) = trainingVec;
    
    cla
    text(0.5,0.5 ,...
        ['Starting Trial ' int2str(currTrial) sprintf('\n') ...
         'Please count the apperances of class' sprintf('\n') classNames(targetClass)], ...
         'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    
    % Show base image for a few seconds before start
    if is_visual
        pause(1)
        cla
        image(flip(diffTrigger, 1))
    end
    
    % Short wait before trial starts
    pause(Utils.Config.preTrialPause);
    
    % Trial - play triggers
    for currTrigger=1:triggersInTrial 
        currClass = trainingVec(currTrigger);
        activateTrigger(trainingSamples, currClass)
        triggersTime(currTrial, currTrigger+1) = now;
        pause(timeBetweenTriggers + rand*Utils.Config.maxRandomTimeBetweenTriggers)  % use random time diff between triggers
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
%     EEG(currTrial, :, :) = recordingBuffer.OutputPort(1).Data'; 
    % add finsh time of trail to allow splitting
    triggersTime(currTrial,(triggersInTrial+1)) = now;    

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
    classNames{1} = 'baseline';
    
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

%     set_param(ampObj, 'Hz', num2str(Hz));           % TODO check how hz is configured in slx

    % Set simulink recording buffer size 
    SampleSizeObj = [usbObj '/Chunk Delay'];        % Todo try to change this name
    set_param(SampleSizeObj,'siz',num2str(eegSampleSize));

    Utils.startSimulation(inf, usbObj);
    open_system(scopeObj);

    recordingBuffer = get_param(SampleSizeObj,'RuntimeObject');
end

function activateVisualTrigger(trainingImages, idx)
    cla
%     imshow(trainingImages{idx})
%     imshow('C:\Ariel\Files\BCI4ALS\gTec4ALS-Team-15\P300\TriggersBank\visual-3-classes\base.jpg', ...
%             'Border','tight')
    imshow(flip(trainingImages{idx}, 1))
end

function activateAudioTrigger(trainingSounds, idx)
    sound(trainingSounds{1, idx}, getSoundFs());
end

function [soundFs] = getSoundFs()
    soundFs = 49920;
end


