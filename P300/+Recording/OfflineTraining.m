function [EEG, fullTrainingVec, expectedClasses, triggersTime, backupTimes] = ... 
    OfflineTraining(timeBetweenTriggers, calibrationTime, pauseBetweenTrials, numTrials, timeBeforeJitter,...
                    numClasses, oddBallProb, triggersInTrial, ...
                    triggerBankFolder, is_visual)
% OfflineTraining - This function is responsible for offline training and
% recording EEG data
% INPUT:
%   - timeBetweenTriggers - time to pause between triggers (doesn't include random time)
%   - calibrationTime - system calibration time in seconds
%   - pauseBetweenTrials - pause time in seconds
%   - numTrials - number of trials in training
%   - timeBeforeJitter - time in seconds before mark jitter happens (jitter blank screen or removing the selected trigger) This is relevant only for visual
%   - numClasses - odd ball classes (e.g., 1-> only one odd ball and baseline)
%   - oddBallProb - probability of the oddball showing, in range [0,1)
%   - triggersInTrial - number of triggers shown in the trial
%   - triggerBankFolder - relative/absolute path to selected trigger bank (folder with images/audio for training)
%   - is_visual - visual or auditory P300

% OUTPUT:
%   - EEG - EEG signal of training. shape: (# trials, # EEG channels, trial sample size) 
%   - fullTrainingVec - Triggers during training. shape: (# trials, trial length)
%   - expectedClasses - class number the subject neeeds to focus on in each trial
%   - triggersTime - times the triggers were activated with last value the dump time of buffer
%   - backupTimes - times before the triggers were activated with last value the time before dumping the buffer


%% Set up recording

[eegSampleSize, recordingBuffer, trainingSamples, diffTrigger, classNames, activateTrigger, fig] = ...
    Recording.RecordingSetup(timeBetweenTriggers, calibrationTime, triggersInTrial, ...
                             triggerBankFolder, timeBeforeJitter, is_visual);

%% Training Setup

fullTrainingVec = ones(numTrials, triggersInTrial);
expectedClasses = zeros(numTrials, 1);
EEG = zeros(numTrials, Utils.Config.eegChannels, eegSampleSize);
triggersTime = zeros(numTrials, (triggersInTrial+1));
backupTimes = zeros(numTrials, (triggersInTrial+1));

%% Training

for currTrial = 1:numTrials
    targetClass = randi(numClasses) + 1;
    expectedClasses(currTrial) = targetClass;
    assert(targetClass > 1 & targetClass <= (numClasses+1), 'Sanity check target class')
   
    [fullTrainingVec(currTrial,:), EEG(currTrial,:,:),triggersTime(currTrial,:), backupTimes(currTrial,:)] = ...
        Recording.SingleTrialRecording(numClasses, oddBallProb, triggersInTrial, is_visual, ...
                                       eegSampleSize, recordingBuffer, ...
                                       Utils.Config.maxRandomTimeBetweenTriggers, Utils.Config.preTrialPause, timeBetweenTriggers, ...
                                       activateTrigger, diffTrigger, trainingSamples, ...
                                       classNames(targetClass), int2str(currTrial));
   
    % End of Trial
    if currTrial ~= numTrials
        endTrialTxt = ['Finished Trial ' int2str(currTrial) sprintf('\n') ...
                     'Pausing for: ' int2str(pauseBetweenTrials) ' seconds before next trial.'];
    else
        endTrialTxt = ['Finished Last Trial - Good Job' sprintf('\n')  ...
            'Processing the data please wait for this window to close'];
    end
    
    set(fig, 'color', 'black');          % imshow removes background color, therefore we need to set it again before showing more text
    Recording.DisplayTextOnFig(endTrialTxt);

    if ~is_visual % Play end sound if needed
        sound(diffTrigger, getSoundFs());
    end

    % Pause between Trials
%     GUIFiles.SuspendRun();
%     input('')
    pause(pauseBetweenTrials)
end        % End trial loop

% Close simulink objects
bdclose all

end





