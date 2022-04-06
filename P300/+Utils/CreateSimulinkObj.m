function [usbObj, scopeObj, impObj, ampObj] = CreateSimulinkObj()
%CREATESIMULINKOBJ Summary of this function goes here
%   Detailed explanation goes here

usbObj          = 'USBamp_offline';
ampObj          = [USBobj '/g.USBamp UB-2016.03.01'];
impObj          = [USBobj '/Impedance Check'];
scopeObj        = [USBobj '/g.SCOPE'];

load_system(['GUIFiles/' USBobj])
set_param(USBobj,'BlockReduction', 'off')

end

