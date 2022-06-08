from dataclasses import dataclass
from typing import Tuple


@dataclass()
class Const:
    eeg_name: Tuple = ('EEG.mat', 'EEG')
    training_sequences: Tuple = ('trainingVector.mat', 'trainingVector')
    training_labels: Tuple = ('trainingLabels.mat', 'trainingLabels')
    training_parameters: Tuple = ('parameters.mat', 'parametersToSave')
    triggers_time: Tuple = ('triggersTimes.mat', 'triggersTimes')
    processed_eeg: Tuple = ('processedEEG', 'processedEEG')
    eeg_classes: Tuple = ('trainingLabels', 'trainingLabels')
    time_pause_between_triggers: str = 'timeBetweenTriggers'
    hz: str = 'Hz'
    num_trials: str = 'numTrials'
    triggers_in_trial: str = 'triggersInTrial'
    search_result_file: str = 'search_results.json'



@dataclass
class Configurations:
    placeholder: str = None


config = Configurations()
const = Const()
