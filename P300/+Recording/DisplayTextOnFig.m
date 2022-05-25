function DisplayTextOnFig(textToDisplay)
% This functions sets all necessary variables to dispaly text on figure
    cla
    text(0.5,0.5, textToDisplay, ...
         'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
end