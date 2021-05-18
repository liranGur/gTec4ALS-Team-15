function KeyBoardGUI(state, handle)


% Rectangle Positions according to the current screen
if length(state.position) == 9
    positions = [57 35.5000000000001 402 186;...
        463 35.5000000000001 402 186;...
        867 35.5000000000001 402 186;...
        56 243.5 402 186;...
        462 243.5 402 186;...
        869 242.5 402 186;...
        55 451.5 402 186;...
        464 452.5 402 186;...
        870 451.5 402 186];
elseif length(state.position) == 6
    positions = [33 170.5 403 188;...
        439 169.5 403 188;...
        847 168.5 403 188;...
        33 377.5 403 188;...
        441 376.5 403 188;...
        846 377.5 403 188];
elseif length(state.position) == 7
    positions = [32 97.5000000000001 405 189;...
        437 96.5000000000001 405 189;...
        845 95.5000000000001 405 189;...
        29 304.5 405 189;...
        439 303.5 405 189;...
        845 302.5 405 189;...
        438 513.5 405 189];
end

% Choose axes to the current handle
axes(handle)
% Load image
img = imread([pwd, '\LoadingPics\' state.screen '.jpg']);
% Draw rectangle
img = insertShape(img,'Rectangle',positions(logical(state.position), :),'LineWidth',6, 'Color', 'red', 'Opacity', 1);
% Show the image
imshow(img)
