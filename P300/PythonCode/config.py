from dataclasses import dataclass
from typing import Tuple


@dataclass()
class Const:
    eeg_name: Tuple = ('EEG.mat', 'EEG')
    training_sequences: Tuple = ('trainingSequences.mat', 'fullTrainingVec')
    training_labels: Tuple = ('trainingLabels.mat', 'expectedClasses')
    training_parameters: Tuple = ('parameters.mat', 'parametersToSave')
    time_pause_between_triggers: str = 'timeBetweenTriggers'
    hz: str = 'Hz'
    start_pause_length: str = 'startingNormalTriggers'
    num_trials: str = 'numTrials'
    trail_length: str = 'trialLength'


@dataclass
class Configurations:
    placeholder: str = None


config = Configurations()
const = Const()
__all__ = ['config', 'const']
