function [EEG, fullTrainingVec, expectedClasses] = ... 
    OfflineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials, numTrials, ...
                    numClasses, oddBallProb, triggersInTrial, baseStartLen, ...
                    Hz, eegChannels, triggerBankFolder)
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

recordingBuffer = setUpRecordingSimulink(Hz, eegSampleSize);

%% Load Train Samples

[endTrialSound, trainingSounds] = GetTriggers(triggerBankFolder, numClasses);

sound_fs = 49920;   % sound frequency

classNames{1} = 'High pitch';
classNames{2} = 'Low Pitch';
classNames{3} = 'What now';



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


fullTrainingVec = ones(numTrials, triggersInTrial);
expectedClasses = zeros(numTrials, 1);
EEG = zeros(numTrials, eegChannels, eegSampleSize);

for currTrial = 1:numTrials
    % Prepare Trial
    trainingVec = Utils.TrainingVecCreator(numClasses, oddBallProb, triggersInTrial, baseStartLen);
    desiredClass = round((numClasses-1)*rand); % $ what this?
    expectedClasses(currTrial) = desiredClass;
    fullTrainingVec(currTrial, : ) = trainingVec;
    text(0.5,0.5 ,...
        ['Starting Trial ' int2str(currTrial) sprintf('\n') 'Please count the apperances of class' classNames(desiredClass)], ...
        'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    pause(preTrialPause)
    
    % Trial - play triggers
    for currTrigger=1:triggersInTrial 
        currClass = trainingVec(currTrigger);
        sound(trainingSounds{1, currClass}, sound_fs);  % find a way to play a sound for specific time
        pause(timeBetweenTriggers)
    end
    
    % End of Trial
    cla
    text(0.5,0.5 ,...
        ['Finished Trial ' int2str(currTrial) sprintf('\n') ...
         'Pausing for: ' int2str(pauseBetweenTrials) ' seconds before next trial.'], ...
         'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    sound(endTrialSound, sound_fs)
    pause(0.5)  % pausing as a safety buffer for final trigger recording in EEG
    EEG(currTrial, :, :) = recordingBuffer.OutputPort(1).Data'; 

    pause(pauseBetweenTrials)
end

close(MainFig)
end


function [endTrailSound, trainingSounds] = GetTriggers(triggerBankFolder, numClasses)
    files = dir(triggerBankFolder);
    for idx = 1:length(files)
        curr_name = files(idx).name;
        if strfind(curr_name, 'end')
            endTrailSound = audioread(strcat(triggerBankFolder,curr_name));
        else if strfind(curr_name, 'base')
                trainingSounds{1} = audioread(strcat(triggerBankFolder,curr_name));
            end
        end
    end
    
    for sample_idx = 2:(numClasses+1)
        for idx = 1:length(files)
            curr_name = files(idx).name;
            if strfind(curr_name,['trigger' int2str(sample_idx-1)])
                trainingSounds{sample_idx} =  audioread(strcat(triggerBankFolder,curr_name));
                break
            end
        end
    end
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