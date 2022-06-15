%% Load no bandpass data
recordingFolder = uigetdir('C:/Subjects/', ...
    'Choose Desired Directory');

load(strcat(recordingFolder,'\EEG.mat'), 'EEG'); %choose non-bandpassed data
EEG_nobdps = EEG;
% trainingVec = load(strcat(recordingFolder,'\trainingVector.mat'), 'trainingVector');
% expectedClasses = load(strcat(recordingFolder,'\trainingLabels.mat'), 'trainingLabels');
% load(strcat(recordingFolder,'\triggersTimes.mat'), 'triggersTimes')

%% Load bandpass data\

EEG_bdps = EEG;
clear EEG
load(strcat(recordingFolder,'\trainingVector.mat'), 'trainingVector');
trainingVec = trainingVector;
load(strcat(recordingFolder,'\trainingLabels.mat'), 'trainingLabels');
expectedClasses = trainingLabels;
load(strcat(recordingFolder,'\triggersTimes.mat'), 'triggersTimes')
load(strcat(recordingFolder,'\meanTriggers.mat'), 'meanTriggers');
meanTrigs = meanTriggers;

%% Setting parameters
Hz = Utils.Config.Hz;
fpass = [1,30];
numTrials = size(EEG_nobdps,1);
numElec = size(EEG_nobdps,2);
N_nobdps = size(EEG_nobdps,3);
N_bdps = size(EEG_bdps,3);
timeVec = (0:N_nobdps-1)/Hz;       %Time vector in seconds for no_bdps data
fig_sz = [1.28764,2.1343,30.8857,14.19049];
Font = struct('axesmall', 13,...
    'axebig', 16,...
    'label', 14,...
    'title', 18); %Axes font size

%% Bandpass
fltrEEG = zeros(size(EEG_nobdps));
for currTrial = 1:numTrials
    for currElec = 1:numElec
        sqzEEG = squeeze(EEG_nobdps(currTrial,currElec,:));   % squeeze specific trial and electrode
        bndpsEEG = bandpass(sqzEEG,fpass,Hz);   % bandpass data of specific trial and electrode
        fltrEEG(currTrial,currElec,:) = bndpsEEG;   % add filtered vectors to array
    end
end

% Visualization - figure for each trial
for currTrial = 1:numTrials
    figure('units' , 'centimeters' , 'position' , fig_sz)
    sgtitle({'EEG after matlab bandpass';'Trial:';num2str(currTrial)},...
        'FontSize', Font.title)
    visEEG = squeeze(fltrEEG(currTrial,:,:));
    %plot each electrode in subplot
    for currElec = 1:numElec
        subplot(4,4,currElec)
        plot(timeVec(5000:end-5000),visEEG(currElec,5000:end-5000))
        title(['Electrode ', num2str(currElec)])
    end
end

%% Check bandpass
%pwelch on nobdps
pwelch_nobdps = zeros(numTrials,numElec,4097);
for iTrial = 1:numTrials
    Y_nobdps = squeeze(EEG_nobdps(iTrial,:,:))';
    N = size(Y_nobdps,1);
    t = (0:N-1)/Hz;
    [Yft_nobdps,f_nobdps] = pwelch(Y_nobdps,[],[],[],Hz);
    pwelch_nobdps(iTrial,:,:) = Yft_nobdps';
end
%pwelch on fltrEEG
pwelch_fltr = zeros(numTrials,numElec,4097);
for iTrial = 1:numTrials
    Y_fltr = squeeze(fltrEEG(iTrial,:,:))';
    N = size(Y_fltr,1);
    t = (0:N-1)/Hz;
    [Yft_fltr,f_fltr] = pwelch(Y_fltr,[],[],[],Hz);
    pwelch_fltr(iTrial,:,:) = Yft_fltr';
end
%pwelch on bdps
pwelch_bdps = zeros(numTrials,numElec,8193);
for iTrial = 1:numTrials
    Y_bdps = squeeze(EEG_bdps(iTrial,:,:))';
    N = size(Y_bdps,1);
    t = (0:N-1)/Hz;
    [Yft_bdps,f_bdps] = pwelch(Y_bdps,[],[],[],Hz);
    pwelch_bdps(iTrial,:,:) = Yft_bdps';
end

%visualization
for iTrial = 1:numTrials
    figure
    sgtitle(['Trial ', num2str(iTrial)]);
    hold on
    %no bandpass subplot
    subplot(3,1,1);
    %plot all electrodes in one subplot
    for iElec = 1:numElec
        plot(f_nobdps(1:end/2), squeeze(pwelch_nobdps(iTrial,iElec,1:end/2)));
        hold on
    end
    title('no bandpass')
    
    %matlab bandpass subplot
    subplot(3,1,2);
    %plot all electrodes in one subplot
    for iElec = 1:numElec
        plot(f_fltr(1:end/2), squeeze(pwelch_fltr(iTrial,iElec,1:end/2)));
        hold on
    end
    title('matlab bandpass')
    ylabel('Power Spectrum')
    
    %simulink bandpass subplot
    subplot(3,1,3);
    %plot all electrodes in one subplot
    for iElec = 1:numElec
        plot(f_bdps(1:end/4), squeeze(pwelch_bdps(iTrial,iElec,1:end/4)));
        hold on
    end
    title('simulink bandpass')
    xlabel('Frequency [Hz]')      
end


%% ERP
L = size(meanTrigs,4);
timeVec = (0:L-1)/Hz; 

currElec = 6;                   %TODO - choose which electrode to see
data = meanTrigs(:,:,currElec,:);
data = squeeze(data);

figure('units' , 'centimeters' , 'position' , fig_sz)
sgtitle(['ERP electrode ' , num2str(currElec)] , 'FontSize' , Font.title)
% EEG graph
for currTrial = 1:numTrials
    expClass = expectedClasses(currTrial);  % the class to focus on
    currTrialData = data(currTrial,:,:);         % shows data from specific trial
    pltData = squeeze(currTrialData);
    [maxy, ind] = max(pltData(expClass,:));     % peak of data
    
    subplot(5,3,currTrial)
    hold on
    title(['Trial ', num2str(currTrial)])
    plot(timeVec, pltData(1,:));           % plots data of baseline class
    plot(timeVec,pltData(expClass,:),...
        timeVec(ind),maxy,'or');    % plots data of expected class and peak
end


