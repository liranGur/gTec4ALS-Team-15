function [trainingVec] = AudioTrainingVecCreator(numClasses, oddBallProb, sequenceLength, baseStartLen)
%TRAININGVECCREATOR Create labels for training vector
% numClasses 
% oddBallProb - probability of the odd ball classes
% sequenceLength - number of signals in trail
% baseStartLen - How much of the first trails will be base trails

trainingVec=ones(sequenceLength,1);
perClassAmount = round(sequenceLength*oddBallProb);

currClass = 2;
currClassAmount = 0;
currClassRuns = 0;
while currClass < numClasses + 2
    for i=baseStartLen+1:sequenceLength 
        if trainingVec(i) ~= 1 
            continue
        end
        prob = rand;
        if prob < oddBallProb 
            currClassAmount = currClassAmount + 1;
            trainingVec(i) = currClass;
        end
        if currClassAmount > perClassAmount
            currClass = currClass + 1;
            currClassAmount = 0;
            currClassRuns = 0;
            break
        end
    end
    
    % In case we didn't create enough labels for the current class
    if currClassAmount >= perClassAmount || currClassRuns == 3 
        currClassAmount = 0;
        currClassRuns = 0;
        currClass = currClass + 1;
    else
        currClassRuns = currClassRuns + 1;
    end
end


end

