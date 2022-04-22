from dataclasses import dataclass
from typing import Tuple


@dataclass()
class Const:
    eeg_name: Tuple = ('EEG.mat', 'EEG')
    training_sequences: Tuple = ('trainingSequences.mat', 'trainingVec')
    training_labels: Tuple = ('trainingLabels.mat', 'expectedClasses')
    training_parameters: Tuple = ('parameters.mat', 'parametersToSave')
    triggers_time: Tuple = ('triggersTime.mat', 'triggersTimes')
    time_pause_between_triggers: str = 'timeBetweenTriggers'
    hz: str = 'Hz'
    num_trials: str = 'numTrials'
    triggers_in_trial: str = 'triggersInTrial'



@dataclass
class Configurations:
    placeholder: str = None


config = Configurations()
const = Const()
