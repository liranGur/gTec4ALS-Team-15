function [trainingVec] = VisualTrainingVecCreator(numClasses, sequenceLength)
    trainingVec = rand(sequenceLength,1);
    trainingVec = round(trainingVec * (numClasses-1)) + 1
end
