function [fig, ax] =  DisplaySetUp()
% Setup display monitor and set up figure for all training

%% Set up monitor
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

%% Set figure

figurePos = monitorPos(choosenMonitor, :);
figure('outerPosition',figurePos);                      % Open figuer in selected monitor

fig = gcf;
ax = gca;

set(ax,'Unit','normalized','Position',[0 0 1 1]);      % Full screen
set(ax,'color', 'black');                              % Background color

set(fig,'menubar','none')                           % Hide menu bar
set(fig,'NumberTitle','off')                        % Hide title bar

% Lock axes limits - This is needed to keeping text in same location after showing images
ax.XLim = [-inf inf];
ax.YLim = [-inf inf];
hold on

end

