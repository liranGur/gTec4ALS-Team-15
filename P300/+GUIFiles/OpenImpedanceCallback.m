function OpenImpedanceCallback(varargin)
% Callback for impedance pushbutton.

% Extract Impedance object string
USBobj = varargin{1,end-1};
IMPobj = varargin{1,end};

set_param(USBobj,'Location',[1300 199 1301 200])

% Open Impedance object
open_system(IMPobj);
end