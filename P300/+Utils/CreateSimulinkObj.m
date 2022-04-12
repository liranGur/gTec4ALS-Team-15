function [usbObj, scopeObj, impObj, ampObj] = CreateSimulinkObj()
%CREATESIMULINKOBJ Summary of this function goes here
%   Detailed explanation goes here

usbObj          = 'USBamp_offline';
ampObj          = [usbObj '/g.USBamp UB-2016.03.01'];
impObj          = [usbObj '/Impedance Check'];
scopeObj        = [usbObj '/g.SCOPE'];

load_system(['GUIFiles/' usbObj])
set_param(usbObj,'BlockReduction', 'off')

end

