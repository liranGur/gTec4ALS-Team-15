function SelectTriggerBankCallback(varargin)
%SELECTTRIGGERBANKCALLBACK Summary of this function goes here
%   Detailed explanation goes here

GUI = varargin{1,end};
selectedBank = uigetdir(strcat(pwd, GUI.bank.UserData), ...
    'Choose Desired Trigger Bank');
GUI.bank.UserData = selectedBank;
end

