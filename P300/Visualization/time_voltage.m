close all; clear; clc;
%% setting single record
recordingFolder = uigetdir('C:/Subjects/', ...
    'Choose Desired Directory');

load(strcat(recordingFolder,'\EEG.mat'), 'EEG')
load(strcat(recordingFolder,'\trainingSequences.mat'), 'trainingVec')
load(strcat(recordingFolder,'\trainingLabels.mat'), 'expectedClasses')
load(strcat(recordingFolder,'\triggersTime.mat'), 'triggersTimes')

%% Setting parameters
Hz = Utils.Config.Hz;
fpass = [1, 30];
L = size(EEG,3);
timeVec = (0:L-1)/Hz;           %Time vector in seconds
fig_sz = [1.28764,2.1343,30.8857,14.19049];
Font = struct('axesmall', 13,...
    'axebig', 16,...
    'label', 14,...
    'title', 18); %Axes font size

%% Bandpass
fltrEEG = zeros(size(EEG));
for currTrial = 1:size(EEG,1)
    for currElec = 1:size(EEG,2)
        sqzEEG = squeeze(EEG(currTrial,currElec,:));   % squeeze specific trial and electrode
        bndpsEEG = bandpass(sqzEEG,fpass,Hz);   % bandpass data of specific trial and electrode
        fltrEEG(currTrial,currElec,:) = bndpsEEG;   % add filtered vectors to array
    end
end

% Visualization - 10 figures (1 for each trial)
for currTrial = 1:size(fltrEEG,1)
    figure('units' , 'centimeters' , 'position' , fig_sz)
    sgtitle(['Trial ',num2str(currTrial)], 'FontSize', Font.title)
    visEEG = squeeze(fltrEEG(currTrial,:,:));
    for iPlot = 1:16
        subplot(4,4,iPlot)
        plot(timeVec,visEEG(iPlot,:))
    end
end
%%


timeVec = timeVec - Utils.Config.preTriggerRecTime;     % Adjust t=0 to be on trigger appearance
trigidx = round(Utils.Config.preTriggerRecTime*Hz)+1;  % time index of trigger appearance

currElec = 6;                   %TODO - choose which electrode to see
data = meanTrigs(:,:,currElec,:);
data = squeeze(data);

elec_name = {'C3', 'C4', 'Cz'};   %TODO - choose electrodes
elec_idx = [5, 9 ,7];             %TODO - choose electrodes indices

%MATLAB R2019b
%
%Creating visualization of the EEG signal with left and right trials
%division: random N trials, spectrogram per elctrode and condition, and the
%condition differnce, power spectrum plot and ERP plot.
%
%data - the amplitude data for all trials and electrodes.
%left_idx , right_idx - the indexis of left \ right trials.
%Fs - Sampling rate.
%f - frequencies vector
%window_sz - window size.
%overlap_sz - overlap size.
%elec_name - electrodes names string.
%imagine_t - time period the subject imagined the hand movement.
%trials_N - number of random trials per condition wanted for ploting.
%max_trials - Maximum numbers of trials to plot per figure.
%Font - Structure of font size.
%
%output- all of the above mentioned graphes, no need to pre assign figure.
%function will create on it own.
%
%--------------------------------------------------------------------------------

%% raw EEG plot

%-----------------------------------------------------
%INPUT:
% - data - values of EEG [in microvolts] across time (to see P300 spikes)
% - trainingVec - time values [in seconds] where oddball was presented
%OUTPUT:
% - time [sec] by amplitude [microvolt] graph of EEG values
% - vertical lines on oddball time points + labels on lines naming the time 
%points [sec]
% - labels on max EEG values after each oddball [sec]
%-----------------------------------------------------

figure('units' , 'centimeters' , 'position' , fig_sz)
sgtitle('P300' , 'FontSize' , Font.title)
hold on
% EEG graph
for currTrial = 1:numTrials
    expClass = expectedClasses(currTrial);  % the class to focus on
    currTrialData = data(currTrial,:,:);         % shows data from specific trial
    pltData = squeeze(currTrialData);
    [maxy, ind] = max(pltData(expClass,:));     % peak of data
    
    subplot(5,2,currTrial)
    title(['Trial ', num2str(currTrial)])
    plot(timeVec, pltData(1,:));           % plots data of baseline class
    plot(timeVec,pltData(expClass,:),...
        timeVec(ind),maxy,'or');    % plots data of expected class and peak
    txt = num2str(timeVec(ind));
    text(timeVec(ind+7),maxy*1.03,txt)  % timestamp label of peak
    xline(timeVec(trigidx),'--k','trigger')     % marks trigger appearance
end
ylabel('Amplitude [\muV]')
xlabel('Time [sec]')
set(gca,'FontSize',Font.axebig)     %Axes font size
legend('Baseline', 'Target')

%% working code
Hz = 512;
L = size(meanTrigs,4);
timeVec = (0:L-1)/Hz; 

currElec = 6;                   %TODO - choose which electrode to see
data = meanTrigs(:,:,currElec,:);
data = squeeze(data);

figure('units' , 'centimeters' , 'position' , fig_sz)
sgtitle('P300' , 'FontSize' , Font.title)
% EEG graph
for currTrial = 1:numTrials
    expClass = expectedClasses(currTrial);  % the class to focus on
    currTrialData = data(currTrial,:,:);         % shows data from specific trial
    pltData = squeeze(currTrialData);
    [maxy, ind] = max(pltData(expClass,:));     % peak of data
    
    subplot(5,2,currTrial)
    hold on
    title(['Trial ', num2str(currTrial)])
    plot(timeVec, pltData(1,:));           % plots data of baseline class
    plot(timeVec,pltData(expClass,:),...
        timeVec(ind),maxy,'or');    % plots data of expected class and peak
%     txt = num2str(timeVec(ind));
%     text(timeVec(ind+7),maxy*1.03,txt)  % timestamp label of peak
%     xline(timeVec(trigidx),'--k','trigger')     NOT RELEVANT % marks trigger appearance
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% previous code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[peaks, peakLocs] = findpeaks(data, Hz,...
    'MinPeakDistance', 0.6);   % finds local maxima
plot(timeVec, data, peakLocs, peaks, 'or')  % plots data and marks local maxima
txt = string(peakLocs);
text(peakLocs, peaks, txt)
ylabel('Amplitude [\muV]')
xlabel('Time [sec]')
set(gca,'FontSize',Font.axebig) %Axes font size

% Oddball vertical lines
verticalLines = arrayfun(@(x) xline(x, '--', 'LabelOrientation', 'horizontal'),...
    XXXX);  % plots a vertical line for each value in "XXXX"

for iLine = 1:length(verticalLines)  % defines label of each line to be its time value
    verticalLines(iLine).Label = XXXX(iLine);
end

% Max EEG labels
% for iSection =       %TODO - subset data into sections and find max for each section

hold off

%% ERP
%Compute ERP
for elec_i = 1:elec_N
    ERP.left.(elec_name{elec_i}) = mean(left_data(:,:,elec_i));
    ERP.right.(elec_name{elec_i}) = mean(right_data(:,:,elec_i));
    ERP.Idle.(elec_name{elec_i}) = mean(idle_data(:,:,elec_i));
end

%Plot ERP
figure('units' , 'centimeters' , 'position' , fig_sz)
sgtitle('ERP' , 'FontSize' , Font.title)
for elec_i = 1:elec_N
    %Left ERP
    subplot(3,1,1)
    hold on
    plot(timeVec , ERP.left.(elec_name{elec_i}))
    title('Left trials')
    ylabel('Amplitude [\muV]')
    xlim([timeVec(1) timeVec(end)])
    set(gca,'FontSize',Font.axebig) %Axes font size.
    
    %Right ERP
    subplot(3,1,2)
    hold on
    plot(timeVec , ERP.right.(elec_name{elec_i}))
    title('Right trials')
    ylabel('Amplitude [\muV]')
    xlabel('Time [Sec]')
    xlim([timeVec(1) timeVec(end)])
    set(gca,'FontSize',Font.axebig) %Axes font size.
    
    %Idle ERP
    subplot(3,1,3)
    hold on
    plot(timeVec , ERP.Idle.(elec_name{elec_i}))
    title('Idle trials')
    ylabel('Amplitude [\muV]')
    xlabel('Time [Sec]')
    xlim([timeVec(1) timeVec(end)])
    set(gca,'FontSize',Font.axebig) %Axes font size.
end
legend(elec_name, 'Position' , [0.921,0.423,0.056,0.1374])

%% Plot ERP difference

for curr = 1:3
    figure('units' , 'centimeters' , 'position' , fig_sz)

    subplot(3,1,1)
    hold on
    plot(timeVec, (ERP.left.(elec_names_diff{curr}{1}) - ERP.left.(elec_names_diff{curr}{2})) )
    title(['Left Trials ', elec_names_diff{curr}{1},'-', elec_names_diff{curr}{2}])
    ylabel('Amplitude [\muV]')
    xlim([timeVec(1) timeVec(end)])
    set(gca,'FontSize',Font.axebig) %Axes font size.

    %Plot ERP difference
    subplot(3,1,2)
    hold on
    plot(timeVec, (ERP.right.(elec_names_diff{curr}{1}) - ERP.right.(elec_names_diff{curr}{2}) ))
    title(['Right Trials ', elec_names_diff{curr}{1},'-', elec_names_diff{curr}{2}])
    ylabel('Amplitude [\muV]')
    xlim([timeVec(1) timeVec(end)])
    set(gca,'FontSize',Font.axebig) %Axes font size.


    %Plot ERP difference
    subplot(3,1,3)
    hold on
    plot(timeVec, (ERP.Idle.(elec_names_diff{curr}{1}) - ERP.Idle.(elec_names_diff{curr}{2})))
    title(['Idle Trials ', elec_names_diff{curr}{1},'-', elec_names_diff{curr}{2}])
    ylabel('Amplitude [\muV]')
    xlim([timeVec(1) timeVec(end)])
    set(gca,'FontSize',Font.axebig) %Axes font size.
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%