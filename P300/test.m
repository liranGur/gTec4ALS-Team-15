function test()
%TEST Summary of this function goes here
%   Detailed explanation goes here


load('recordingFolder\100\24-5_bandpass\trainingLabels.mat')
load('recordingFolder\100\24-5_bandpass\EEG.mat')
load('recordingFolder\100\24-5_bandpass\targets.mat')
load('recordingFolder\100\24-5_bandpass\trainingVector.mat')
load('recordingFolder\100\24-5_bandpass\trainingLabels.mat')
load('recordingFolder\100\24-5_bandpass\triggersTimes.mat')


preTriggerRecTime = -0.2;
triggerWindowTime = 0.8;
downSampleRate = 20;
[splitEEG, meanTriggers, subtractedMean, processedEEG] = preprocessing(EEG, triggersTimes, trainingVector, ...
                                         preTriggerRecTime, triggerWindowTime, downSampleRate);
[trainData, targets] = Models.processedDataTo2dMatrixMeanChannels(processedEEG, trainingLabels, 1);
% save('recordingFolder\100\24-5_bandpass\data_test.mat', 'trainData')
% save('recordingFolder\100\24-5_bandpass\data_target.mat', 'targets')


%% Raw EEG

%plot raw EEG - mean on channels single trial
trailToPlot = 8;
figure('Name', strcat('raw eeg for trial ', int2str(trailToPlot), ' mean on channels'))
plot(squeeze(mean(squeeze(EEG(trailToPlot,:,:)),1)))

%% MeanTriggers


figure()
trial=10;
for i=1:100
    hold on
    if trainingVector(trial,i) == 2
        plot(squeeze(splitEEG(1,i,9,:)), 'b')
    end
    if trainingVector(trial,i) == 3
        plot(squeeze(splitEEG(1,i,9,:)), 'r')
    end
end

%plot meanEEG for targets - mean on channels
figure('Name', 'EEG - Mean Triggers (no preprocess) - only on targets of train - mean on channels')
for i=1:15
    currCls = trainingLabels(i);
    toPlot = squeeze(mean(squeeze(meanTriggers(i,currCls, :, :)),1));
    subplot(4,4,i)
    plot(toPlot)
    title(currCls)
end


% plot mean data - mean on channels - all classes
diffTrial = 10;
figure('Name', 'EEG - Mean Triggers (no preprocess) All classes')
for trial=diffTrial:diffTrial+2
    for cls=1:4
        toPlot = squeeze(mean(squeeze(meanTriggers(trial, cls, :, :)),1));
        subplot(3,4,(trial-diffTrial)*4 + cls)
        plot(toPlot)
        if trainingLabels(trial) == cls
            title(strcat('target ', int2str(trial), '----', int2str(cls)))
        else
            title(strcat(int2str(trial), '----', int2str(cls)))
        end
    end
end


% subtract baseline
subtractedMean = zeros(4,16,308);
trialIdx=3;
for cls=1:4
    for channel=1:16
        toSubtract = mean(meanTriggers(trialIdx,cls,channel,1:100));
        subtractedMean(cls,channel,:) = meanTriggers(trialIdx,cls,channel,:) - toSubtract;
    end
end

figure()
for cls=1:4
    for chan=1:16
        subplot(4,16,(cls-1)*4+chan)
        plot(squeeze(subtractedMean(cls,channel,:)))
    end
end

figure()
for chan=1:16
    for cls=[1 3]
        subplot(4,4,chan)
        hold on
        plot(squeeze(subtractedMean(cls,chan,:)))
    end
end

%% Split EEG

%Plot split trials - mean on channels
firstTrigger = 1;
trialIdx = 1;
figure('Name', 'Split EEG with mean on channels (Not mean of classes in trial)')
for i=firstTrigger:firstTrigger+15
    currTargetClass = trainingLabels(trialIdx);
    toPlot = squeeze(mean(squeeze(splitEEG(trialIdx,i,:,:)),1));
    subplot(4,4,i-firstTrigger+1)
    plot(toPlot)
    if trainingVector(trialIdx, i) == currTargetClass
        title('target')
    else
        title(strcat('non-target - ', int2str(i)))
    end
    
end

%% subtracted Mean

% plot all channles 3 figures one for each class
trialIdx = 9;
for cls=2:4
    figure('Name', strcat('subtracted mean class: ', int2str(cls)))
    for chan = 1:16
        subplot(4,4,chan)
        plot(squeeze(subtractedMean(trialIdx,cls, chan, :)))
        title(int2str(chan))
    end
end

%% Processed EEG

% Processed EEG
figure('Name', 'Fully Processed EEG - only target class')
for i=1:4
    for cls=1:4
        toPlot = squeeze(mean(squeeze(processedEEG(i,cls, :, :)),1));
        subplot(4,4,(i-1)*4 + cls)
        plot(toPlot)
        if cls == trainingLabels(i);
            title('Target')
        else
            title('Non Target')
        end
    end
end


%plot targets in processedEEG - mean on channels
figure('Name', 'Fully Processed EEG - only target class')
for i=1:15
    currCls = trainingLabels(i);
    toPlot = squeeze(mean(squeeze(processedEEG(i,currCls, :, :)),1));
    subplot(4,4,i)
    plot(toPlot)
    title(currCls)
end

%% Train Data

% plot trainData
figure('Name', 'Model Train data')
for i=1:16
    subplot(4,4,i)
    plot(trainData(i,:))
    title(targets(i))
end

%plot only targets in train data
figure('Name', 'Model Train data only target class')
target_train_data = trainData(targets==1, :);
for i=1:15
    subplot(4,4,i)
    plot(target_train_data(i,:))
end

%plot only non-targets in train data
figure('Name', 'Model Train data only NON target class')
target_train_data = trainData(targets==0, :);
for i=1:16
    subplot(4,4,i)
    plot(target_train_data(i,:))
end

%% Ignore

% 
%     if 1 == 2
%         func = @(a,b) t1(a,b, 9);
%     else
%         func = @(a,b) t2(a,b);
%     end
% 
%     x = func(1,3);
%     display('==== ')
%     display(x);
end


function [z] = t1(a,b, c)
    display('1');
    z = a + b + c;
end

function [z] = t2(a,b)
    display('2');
    z = b - a;
end
