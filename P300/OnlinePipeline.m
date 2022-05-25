function [] = OnlinePipeline()
%OnlinePipeline - this is the online inference pipeline for communication
%

%% Load model & record parameters

%%% Here we allow selecting a user directory or a specific model directory
modelFolder = uigetdir('G:\.shortcut-targets-by-id\1EX7NmYYOTBYtpFCqH7TOhhm4mY31oi1O\P300-Recordings', ...
    'Choose Desired Directory for Saving Recordings');

% if the path of the folder won't contain allModels we assume that only a
% subject folder was selecte thus we need to select a specific model fodler
if ~strfind(modelFolder, Utils.Config.modelDirName)
    modelFolder = selectModelByAcc(modelFolder);
end

if stfind(modelFolder,'SVM')
    model = load(strcat(modelFolder, 'model.mat'));
    model = model.finalModel;
elseif stfind(modelFolder,'LDA')
    error('to do')
else
    error('Online training unknwon model type')
end


%% online parameters

offlineParameters = load(strcat(modelFolder, 'parameters.mat'));
offlineParameters = offlineParameters.parametersToSave;

numClasses = offlineParameters.numClasses;
timeBetweenTriggers = offlineParameters.timeBetweenTriggers;
beforeJitterTime = offlineParameters.beforeJitterTime;
startingNormalTriggers = offlineParameters.startingNormalTriggers;
oddBallProb = offlineParameters.oddBallProb;
triggersInTrial = offlineParameters.triggersInTrial;
triggerBankFolder = offlineParameters.triggerBankFolder;

% This is a heuristic that needs to be changed to allow shorter online phase than oflline training pahse
if ceil(1/oddBallProb)*numClasses >= 2*triggersInTrial 
    triggersInTrial = ceil(1/oddBallProb)*numClasses + startingNormalTriggers;
end

%% online Train

%% predict


end


function [folderPath] = selectModelByAcc(baseDir)
    modelsDir = [baseDir '\' Utils.Config.modelDirName '\'];
    dirs = ls(modelsDir);
    sortedDirs = sort(dirs);
    sortedDirs = sortedDirs(2:end, :);   % remove . & ..
    bestModelDir = sortedDirs(1:,:);
    folderPath = [modelsDir bestModelDir '\']
end