function OpenImpedanceCallback(varargin)
% Callback for impedance pushbutton.

% Extract Impedance object string
[usbObj, scopeObj, impObj, ampObj] = Utils.CreateSimulinkObj();

set_param(usbObj,'Location',[1300 199 1301 200])

open_system(impObj);
open_system(scopeObj);
Utils.startSimulation(inf, usbObj);

end