function [usbObj, scopeObj, impObj, ampObj] = CreateSimulinkObj()
%CREATESIMULINKOBJ Creates the simulink object and loads the simulink
%systems
% 
%  OUTPUT:
%   - usbObj - simulink usb object - used for getting the different
%   simulink model parts
%   - scopeObje - simulink scope object
%   - impObj - simulink impedance object
%   - ampObj - simulink amplifier object

usbObj          = 'USBamp_offline';
ampObj          = [usbObj '/g.USBamp UB-2016.03.01'];
impObj          = [usbObj '/Impedance Check'];
scopeObj        = [usbObj '/g.SCOPE'];

load_system(['GUIFiles/' usbObj])
set_param(usbObj,'BlockReduction', 'off')

end

