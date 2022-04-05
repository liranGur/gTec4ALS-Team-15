function SelectTriggerBankCallback(varargin)
%SELECTTRIGGERBANKCALLBACK Summary of this function goes here
%   Detailed explanation goes here

triggerBank = varargin{1,end};
selectedBank = uigetdir(strcat(pwd, 'TriggersBank'), ...
    'Choose Desired Trigger Bank');
triggerBank{1} = selectedBank;

end

