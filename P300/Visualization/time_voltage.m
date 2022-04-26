close all; clear; clc;
%% setting single record
recordingFolder = uigetdir('C:/Subjects/', ...
    'Choose Desired Directory');

load(strcat(recordingFolder,'\EEG.mat'), 'EEG')
load(strcat(recordingFolder,'\trainingVec.mat'), 'trainingVec')

%% Setting parameters
data = EEG(1,:,12);               %TODO - change to desired data
Fs = 512;
elec_name = {'C3', 'C4', 'Cz'};   %TODO - choose electrodes
elec_idx = [5, 9 ,7];             %TODO - choose electrodes indices
Font = struct('axesmall', 13,...
    'axebig', 16,...
    'label', 14,...
    'title', 18); %Axes font size
L = size(data,2);
timeVec = (0:L-1)/Fs; %Time vector in seconds for all samples in signal
trainingVec = [timeVec(1)+0.3:0.6:timeVec(end)];  % example to see the
% code works
fig_sz = [1.28764,2.1343,30.8857,14.19049];

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
sgtitle('raw EEG' , 'FontSize' , Font.title)
hold on
% EEG graph
[peaks, peakLocs] = findpeaks(data, Fs,...
    'MinPeakDistance', 0.6);   % finds local maxima
plot(timeVec, data, peakLocs, peaks, 'or')  % plots data and marks local maxima
txt = string(peakLocs);
text(peakLocs, peaks, txt)
ylabel('Amplitude [\muV]')
xlabel('Time [sec]')
set(gca,'FontSize',Font.axebig) %Axes font size

% Oddball vertical lines
verticalLines = arrayfun(@(x) xline(x, '--',...
    'LabelOrientation', 'horizontal'),...
    trainingVec);  % plots a vertical line for each value in "trainingVec"

for iLine = 1:length(verticalLines)  % defines label of each line to be its time value
    verticalLines(iLine).Label = trainingVec(iLine);
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
