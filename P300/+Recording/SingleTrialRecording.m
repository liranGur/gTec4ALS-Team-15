function [trainingVec, trialEEG, triggersTime, backupTimes] = SingleTrialRecording(...
                                                        numClasses, oddBallProb, triggersInTrial, is_visual, ...
                                                        eegSampleSize, recordingBuffer, ...
                                                        maxRandomTimeBetweenTriggers, preTrialPause, timeBetweenTriggers , ...
                                                        activateTrigger, diffTrigger, trainingSamples, ...
                                                        expectedClassName, currTrialStr ...
                                                        )
% 
% 
% 
%% Prepare Trial
    triggersTime = zeros(triggersInTrial + 1, 1);
    backupTimes = zeros(triggersInTrial + 1, 1);
    trialEEG = zeros(Utils.Config.eegChannels, eegSampleSize);
    
    trainingVec = Utils.TrainingVecCreator(numClasses, oddBallProb, triggersInTrial, is_visual);
    assert(all(trainingVec <= (numClasses + 1)), 'Sanity check training Vector')
    Recording.DisplayTextOnFig(['Starting Trial ' currTrialStr sprintf('\n') ...
                                     ' Please count the apperances of the class ' expectedClassName]);
     
%% Pre-trial
    % Show base image for a few seconds before trial starts
    if is_visual
        pause(preTrialPause);
        Recording.DispalyImageWrapper(diffTrigger)
    end

    pause(preTrialPause);    

%% Recording Loop

    for currTrigger=1:triggersInTrial 
        currClass = trainingVec(currTrigger);
        [pre_time, currTriggerTime] = activateTrigger(trainingSamples, currClass);
        triggersTime(currTrigger) = currTriggerTime;
        backupTimes(currTrigger) = pre_time;
        pause(timeBetweenTriggers + rand * maxRandomTimeBetweenTriggers)  % use random time diff between triggers
    end
    
%% Finish trial

    pause(Utils.Config.pauseBeforeDump);
    backupTimes(triggersInTrial+1) = posixtime(datetime('now'));
%     trialEEG(currTrial, :, :) = recordingBuffer.OutputPort(1).Data'; 
    triggersTime(triggersInTrial+1) = posixtime(datetime('now'));
    
end

