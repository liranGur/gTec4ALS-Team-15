function [EEG, fullTrainingVec, expectedClasses] = ...  % TODO return EEG and ??
    OfflineTrainingP(timeBetweenTriggers, calibrationTime, pauseBetweenTrails, numTrails, ...
                     numClasses, oddBallProb, sequenceLength, baseStartLen, ...
                     Hz, recordingFolder)
% amsel TODO
% 1) replace Hz and ??? with USBObj that will be received as parameter
% How can I record overlapping smaples - use 2 overlapping buffers / do it
% manually

%% Set up Simulink
USBobj          = 'USBamp_offline';
AMPobj          = [USBobj '/g.USBamp UB-2016.03.01'];
IMPobj          = [USBobj '/Impedance Check'];
SampleSizeObj   = [USBobj '/Sample Size'];
scopeObj        = [USBobj '/g.SCOPE'];          % amsel TODO WHAT Is THIS

% RestDelayobj    = [USBobj '/Resting Delay'];
% ChunkDelayobj   = [USBobj '/Chunk Delay'];

trailTime = baseStartLen + sequenceLength*timeBetweenTriggers;
eegSampleSize = Hz*trailTime;                      
set_param(SampleSizeObj,'siz',eegSampleSize);

% open Simulink
open_system(['Utillity/' USBobj])
set_param(USBobj,'BlockReduction', 'off')       % amsel TODO WHAT Is THIS

Utillity.startSimulation(inf, USBobj);
open_system(scopeObj);

recordingBuffer = get_param(SampleSizeObj,'RuntimeObject');
eegChannels = 16;



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
fullTrainingVec = ones(numTrails, sequenceLength);
expectedClasses = zeros(numTrails, 1);
EEG = zeros(numTrails, eegChannels, eegSampleSize);
for currTrail = 1:numTrails
    % Prepare Trail
    trainingVec = Utils.TrainingVecCreator(numClasses, oddBallProb, sequenceLength, baseStartLen);
    desiredClass = round((numClasses-1)*rand);
    expectedClasses(currTrail) = desiredClass;
    fullTrainingVec(currTrail, : ) = trainingVec;
    text(0.5,0.5 ,...
        ['Starting Trail ' int2str(currTrail) sprintf('\n') 'Please count the apperances of class' classNames(desiredClass)], ...
        'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    pause(startingNormalTriggers)
    
    %Trail
    for currSeq=1:sequenceLength 
        currClass = trainingVec(currSeq);
        sound(trainingSound{1, currClass}, sound_fs);  % find a way to play a sound for specific time
        pause(timeBetweenTriggers)
        EEG(currTrail, :, :) = recordingBuffer.OutputPort(1).Data';
    end
    
    % End of Trail
    cla
    text(0.5,0.5 ,...
        ['Finished Trail ' int2str(currTrail) sprintf('\n') ...
         'Pausing for: ' int2str(pauseBetweenTrails) ' seconds before next trail.'], ...
         'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    sound(endTrailSound, sound_fs)
    pause(pauseBetweenTrails)
    
end


%% Save Results
save(strcat(recordingFolder, 'trainingSequences.mat'), 'fullTrainingVec');
save(strcat(recordingFolder, 'EEG.mat'), 'EEG');
save(strcat(recordingFolder, 'trainingLabels.mat'), 'expectedClasses');
parametersToSave = struct('timeBetweenTriggers', timeBetweenTriggers, ...
                           'calibrationTime', calibrationTime, ...
                           'pauseBetweenTrails',pauseBetweenTrails, ...
                           'numTrails', numTrails, ... 
                           'startingNormalTriggers', startingNormalTriggers, ...
                           'numClasses', numClasses, ...
                           'oddBallProb', oddBallProb, ...
                           'sequenceLength', sequenceLength, ...
                           'baseStartLen', baseStartLen, ...
                           'Hz', Hz, ...
                           'trailTime', trailTime);
save(strcat(recordingFolder, 'parameters.mat'), 'parametersToSave')

close(MainFig)