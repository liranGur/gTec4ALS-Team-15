function [trainingSamples, diffTrigger, classNames] = LoadTrainingSamples(triggerBankFolder, is_visual)
% loadTrainingSamples - load images / audio for training
%
% INPUT:
%   triggerBankFolder - folder to load triggers from
%   is_visal - 
% 
% OUTPUT:
%   trainingSamples - struct of all the triggers loaded according to their class index
%   diffTrigger - This is an image/sound that isn't part of the triggers but is needed for training.
%       In visual mode this is the basic shapes without selecting any (yellow rectangle)
%       In auditory mode this is the end trial sound
%   classNames - struct of each trigger name


    % This function does the actual loading from flie
    function [trigger] = load_trigger_func(varargin)
       path =  varargin{1,end};
       path = path{1};
       if is_visual
           trigger = imread(path);
       else
           trigger = audioread(path);
       end
    end


    files = dir(triggerBankFolder);
    for i = 1:length(files)
        file_names{i} = files(i).name;
    end
    file_names = sort(file_names);
    file_names = file_names(3:length(file_names));      % remove . & .. from file_names
    classNames{1} = 'baseline';
    
    diffTrigger = load_trigger_func(strcat(triggerBankFolder, '\', file_names(1)));
    for i=2:length(file_names)
        trainingSamples{i-1} = load_trigger_func(strcat(triggerBankFolder, '\', file_names(i)));
        classNames{i-1} = getClassNameFromFileName(file_names(i));
    end
end


function [name] = getClassNameFromFileName(file_name)
% getClassNameFromFileName - extract the class name from the file name.
% !!! All trigger file names should follow this template:  trigger<idx>_<name>.<file type> !!!
    file_name = file_name{1};
    start_loc = strfind(file_name, '_');
    start_loc = start_loc(1);
    end_loc = strfind(file_name, '.');
    end_loc = end_loc(1);
    name = file_name(start_loc:(end_loc-1));
end