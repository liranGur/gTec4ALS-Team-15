function DisplaySetUp()
%DISPLAYSETUO Summary of this function goes here
%   Detailed explanation goes here
% Checking monitor position and number of monitors
monitorPos = get(0,'MonitorPositions');
monitorN = size(monitorPos, 1);
% Which monitor to use TODO: make a parameter
choosenMonitor = 1;
% If no 2nd monitor found, use the main monitor
if choosenMonitor < monitorN
    choosenMonitor = 1;
    disp('Another monitored is not detected, using main monitor')
end
% Get choosen monitor position
figurePos = monitorPos(choosenMonitor, :);

% Open full screen monitor
figure('outerPosition',figurePos);

% get the figure and axes handles
MainFig = gcf;
hAx  = gca;

% set the axes to full screen
set(hAx,'Unit','normalized','Position',[0 0 1 1]);
% hide the toolbar
set(MainFig,'menubar','none')
% to hide the title
set(MainFig,'NumberTitle','off');
% Set background color
set(hAx,'color', 'black');
% Lock axes limits
hAx.XLim = [0, 1];
hAx.YLim = [0, 1];
hold on

end

