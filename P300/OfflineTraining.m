function [EEG, fullTrainingVec, expectedClasses] = ... 
    OfflineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials, numTrials, ...
                    numClasses, oddBallProb, trialLength, baseStartLen, ...
                    USBobj, Hz, eegChannels)
% OfflineTraining - This function is responsible for offline training and
% recording EEG data
% INPUT:
%   - timeBetweenTriggers - in seconds
%   - calibrationTime - system calibration time in seconds
%   - pauseBetweenTrials - in seconds
%   - numTrials
%   - numClasses - odd ball classe (e.g., 1-> only one odd ball and baseline)
%   - oddBallProb - in [0,1]
%   - trialLength - number of triggers in a trial
%   - baseStartLen - number baseline triggers in the start of each trial
%   - USBobj - Simulink object
%   - Hz - EEG recording frequency
%   - eegChannels - number of channels recorded (i.e., number of electordes)

% OUTPUT:
%   EEG - EEG signal of training. shape: (# trials, # EEG channels, trial sample size) 
%   fullTrainingVec - Triggers during training. shape: (# trials, trial length)
%   expectedClasses - Class the subject neeeds to focus on in each trial
%

%% Setup & open Simulink

% Set simulink recording buffer size
SampleSizeObj = [USBobj '/Sample Size'];
trailTime = trialLength*timeBetweenTriggers + 1; % Add 1 seconds as a safety buffer at the begingin of trail EEG data
eegSampleSize = Hz*trailTime;                      
set_param(SampleSizeObj,'siz',eegSampleSize);


scopeObj = [USBobj '/g.SCOPE'];
open_system(['Utillity/' USBobj])
set_param(USBobj,'BlockReduction', 'off')       % amsel TODO WHAT Is THIS

Utillity.startSimulation(inf, USBobj);
open_system(scopeObj);

recordingBuffer = get_param(SampleSizeObj,'RuntimeObject');

%% Load Train Samples
endTrailSound = audioread('./Sounds/');
trainingSound{1} = audioread('./Sounds/'); % base sound
for i= 1:numClasses
    trainingSound{i+1} = audioread(strcat('../Sounds/sound_',int2str(i)));
end
sound_fs = 49920;   % sound frequency

classNames{1} = 'High pitch';
classNames{2} = 'Low Pitch';
classNames{3} = 'What now';

%% Display Setup
% Checking monitor position and number of monitors
monitorPos = get(0,'MonitorPositions');
monitorN = size(monitorPos, 1);
% Which monitor to use TODO: make a parameter
choosenMonitor = 1;
% If no 2nd monitor found, use the main monitor
if choosenMonitor < monitorN
    choosenMonitor = 1;
    disp('Another monitored is not detected, using main monitor')
end
% Get choosen monitor position
figurePos = monitorPos(choosenMonitor, :);

% Open full screen monitor
figure('outerPosition',figurePos);

% get the figure and axes handles
MainFig = gcf;
hAx  = gca;

% set the axes to full screen
set(hAx,'Unit','normalized','Position',[0 0 1 1]);
% hide the toolbar
set(MainFig,'menubar','none')
% to hide the title
set(MainFig,'NumberTitle','off');
% Set background color
set(hAx,'color', 'black');
% Lock axes limits
hAx.XLim = [0, 1];
hAx.YLim = [0, 1];
hold on

%% Callibrate System

% Show a message that declares that training is about to begin
text(0.5,0.5 ,...
    ['System is calibrating.' sprintf('\n') 'The training session will begin shortly.'], ...
    'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
pause(calibrationTime)

% Clear axis
cla

%% Record trails
preTrialPause = 2;

fullTrainingVec = ones(numTrials, trialLength);
expectedClasses = zeros(numTrials, 1);
EEG = zeros(numTrials, eegChannels, eegSampleSize);
for currTrail = 1:numTrials
    % Prepare Trail
    trainingVec = Utils.TrainingVecCreator(numClasses, oddBallProb, trialLength, baseStartLen);
    desiredClass = round((numClasses-1)*rand);
    expectedClasses(currTrail) = desiredClass;
    fullTrainingVec(currTrail, : ) = trainingVec;
    text(0.5,0.5 ,...
        ['Starting Trail ' int2str(currTrail) sprintf('\n') 'Please count the apperances of class' classNames(desiredClass)], ...
        'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    pause(preTrialPause)
    
    %Trail
    for currSeq=1:trialLength 
        currClass = trainingVec(currSeq);
        sound(trainingSound{1, currClass}, sound_fs);  % find a way to play a sound for specific time
        pause(timeBetweenTriggers)
    end
    
    EEG(currTrail, :, :) = recordingBuffer.OutputPort(1).Data';  
    % End of Trail
    cla
    text(0.5,0.5 ,...
        ['Finished Trail ' int2str(currTrail) sprintf('\n') ...
         'Pausing for: ' int2str(pauseBetweenTrials) ' seconds before next trail.'], ...
         'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    sound(endTrailSound, sound_fs)
    pause(pauseBetweenTrials)
    
end

close(MainFig)