function [is_visual, trialLength, numClasses, subId, numTrials, timeBetweenTriggers, oddBallProb, ...
    calibrationTime, pauseBetweenTrials, triggerBank, timeBeforeJitter] = ParametersGui()
%ParametersGui - Display the parameters gui for offline recording

% location parameters
editor_width = 0.1;
editor_height = 0.1;
text_width = 0.15;
text_height = 0.1;
lables_col_x_pos = [0.05 0.35 0.65];
editor_col_x_pos = [0.2 0.5 0.8];
labels_row_y_pos = [0.65 0.45 0.25];
editor_row_y_pos = [0.67 0.47 0.27];

%% Others
GUI.fh = figure('units','normalized',...
    'position',[0.2 0.3 0.5 0.4],...
    'menubar','none',...
    'name','Parameter setting',...
    'numbertitle','off',...
    'resize','off');

% Set title text
GUI.title = uicontrol('style','text',...
    'unit','normali  zed',...
    'position',[0.2 0.7 0.6 0.2],...
    'string','Set parameters for your system');

% Set confirmation button
GUI.confirm = uicontrol('style','push',...
    'unit','normalized',...
    'position',[0.72 0.08 0.2 0.1],...
    'string','Ok');

% Set impedance button
GUI.Imp = uicontrol('style','push',...
    'unit','normalized',...
    'position',[0.125 0.08 0.2 0.1],...
    'string','Impedance');

% Select trigger bank folder
GUI.bank = uicontrol('style','push',...
    'unit','normalized',...
    'position',[0.42 0.08 0.2 0.1],...
    'string','Select Triggers');


%% Configurations

% Set subID text
[GUI.subIDtxt, GUI.subID] = LabelEditorCreator(...
    [lables_col_x_pos(1) labels_row_y_pos(1) text_width text_height], 'Subject ID', ...
    [editor_col_x_pos(1) editor_row_y_pos(1) editor_width editor_height], '500');

%number of classes
[GUI.nClstxt, GUI.nCls] = LabelEditorCreator(...
    [lables_col_x_pos(1) labels_row_y_pos(2) text_width text_height], '# of classes:',...
    [editor_col_x_pos(1) editor_row_y_pos(2) editor_width editor_height], '2');

%system calibration time
[GUI.calibTxt, GUI.calibrationTime] = LabelEditorCreator(...
    [lables_col_x_pos(1) labels_row_y_pos(3) text_width text_height], 'Calibration Time (sec):', ...
    [editor_col_x_pos(1) editor_row_y_pos(3) editor_width editor_height], '30');

%time between triggers 
[GUI.timeBetweenTriggersTxt, GUI.timeBetweenTriggers] = LabelEditorCreator(...
    [lables_col_x_pos(2) labels_row_y_pos(1) text_width text_height], ['Time between ', sprintf('\n'), 'triggers(sec):'],...
    [editor_col_x_pos(2) editor_row_y_pos(1) editor_width editor_height],'0.25,0.2');

%Oddball Probability
[GUI.trialLengthTxt, GUI.trialLength] = LabelEditorCreator(...
    [lables_col_x_pos(2) labels_row_y_pos(2) text_width text_height], ['Trail Length', sprintf('\n'), '(# of triggers):'], ...
    [editor_col_x_pos(2) editor_row_y_pos(2) editor_width editor_height], '30');

%pause between trials
[GUI.pauseBetweenTrialsTxt, GUI.pauseBetweenTrials] = LabelEditorCreator(...
    [lables_col_x_pos(2) labels_row_y_pos(3) text_width text_height], ['Pause Between', sprintf('\n'), 'Trails (sec):'], ...
    [editor_col_x_pos(2) editor_row_y_pos(3) editor_width editor_height], '10');

%num of trials text
[GUI.nTrialtxt,GUI.nTrial] = LabelEditorCreator(...
    [lables_col_x_pos(3) labels_row_y_pos(1) text_width text_height],'# of trials', ...
    [editor_col_x_pos(3) editor_row_y_pos(1) editor_width editor_height],'10');


% Oddball Probability
[GUI.oddBallProbTxt, GUI.oddBallProb] = LabelEditorCreator(...
    [lables_col_x_pos(3) labels_row_y_pos(2) text_width text_height],'Oddball probability:', ...
    [editor_col_x_pos(3) editor_row_y_pos(2) editor_width editor_height],'0.14');


% Set visual / auditory options
GUI.avText = uicontrol('style','text',...
    'unit','normalized',...
    'position',[lables_col_x_pos(3) labels_row_y_pos(3) text_width editor_height],...
    'string','Train Mode:');
GUI.avType    = uicontrol('style','popupmenu',...
    'unit','normalized',...
    'position',[editor_col_x_pos(3) (editor_row_y_pos(3)-0.02) editor_width editor_height],...
    'string',['Visual  ';'Auditory';]);

%% Callbacks
% triggerBank{1}= strcat(pwd, );
GUI.bank.UserData = '.\TriggersBank\visual-2-white';
set(GUI.Imp,'callback',{@GUIFiles.OpenImpedanceCallback});
set(GUI.bank,'callback',{@GUIFiles.SelectTriggerBankCallback, GUI})

% This function is needed because I couldn't call uiresume as a direct
% callback from the GUI button
    function releaseGui(varargin)
        uiresume(GUI.fh)
    end

set(GUI.confirm, 'callback', {@releaseGui});
%% Wait for user reaction
uiwait(GUI.fh);

%% Extract user input parameters

selectedMode = GUI.avType.Value;
is_visual    = selectedMode == 1;
trialLength = str2double(GUI.trialLength.String);
numClasses = str2double(GUI.nCls.String);
subId = str2double(GUI.subID.String);
numTrials = str2double(GUI.nTrial.String);

oddBallProb = str2double(GUI.oddBallProb.String);
calibrationTime = str2double(GUI.calibrationTime.String);
triggerBank = GUI.bank.UserData;
pauseBetweenTrials = str2double(GUI.pauseBetweenTrials.String);

pauseResponse = GUI.timeBetweenTriggers.String;
splitIdx = strfind(pauseResponse, ',');
timeBetweenTriggers = str2double(pauseResponse(1:splitIdx-1));
timeBeforeJitter = str2double(pauseResponse(splitIdx+1:length(pauseResponse)));

close(GUI.fh);

end

function [label, editor] = LabelEditorCreator(label_pos, lablel_text, editor_pos, editor_text)
% LabelEditorCreator - create label and editor text field

label = uicontrol('style','text',...
                  'unit','normalized',...
                  'position',label_pos,...
                  'string', lablel_text);
              
editor = uicontrol('style','edit',...
                   'unit','normalized',...
                   'position',editor_pos, ...
                   'string',editor_text);

end
