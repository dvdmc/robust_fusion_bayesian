#!/usr/bin/env python
# In this script we gather the confusion matrix from
# the results of the experiments, aggregate them and
# get the metrics.
# The results are assumed to have the following structure:
# <experiments_path>
#     |
#     |__ <sequence_name/map>
#             |
#             |__ <experiment_name/method>
#
# Be aware that you should provide the path to the experiments folder
# and not to the sequence_name/map folder.
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import tyro

@dataclass
class ConfusionMatrix:
    """
    Class to store the confusion matrix of a step
    """

    step: int
    """ Step of the confusion matrix """
    path: Path
    """ Path to the confusion matrix file """
    matrix: np.ndarray
    """ Confusion matrix """

    def save(self, path: Path):
        """
        Save the confusion matrix to a file

        Args:
            path: Path to save the confusion matrix
        """
        file_name = self.path / (f"{self.step}:04d.txt")
        np.savetxt(file_name, self.matrix, fmt="%d")

@dataclass
class ExperimentData:
    """
    Class to store the data of an experiment
    """

    map_name: str
    """ Name of the map/sequence """
    name: str
    """ Name of the experiment """
    timestamp: str
    """ Timestamp of the experiment """
    path: Path
    """ Path to the experiment folder """
    confusion_matrices: List[ConfusionMatrix]
    """ List of all the confusion matrices organied by step """
    data: dict
    """ Dictionary with the data of the experiment """

    def save(self, path: Path):
        """
        Save the experiment data to files

        Args:
            path: Path to save the experiment data
        """
        experiment_save_path = path / self.map_name / self.name
        file_name = experiment_save_path / "config.txt"
        with open(file_name, "w") as f:
            for key, value in self.data.items():
                if key == "confusion_matrices":
                    continue
                f.write(f"{key}: {value}\n")

        for confusion_matrix in self.confusion_matrices:
            confusion_matrix.save(experiment_save_path)

def gather_experiment_data(
    experiments_path: Path, output_path: Optional[Path] = None
) -> Dict[str, Dict[str, ExperimentData]]:
    """
    Gather the data from the experiments

    Args:
        experiments_path: Path to the experiments folder
        output_path: Path to the output folder

    Returns:
        Dictionary with the data of the experiments
    """
    # List of maps obtained from listdir
    maps = os.listdir(experiments_path)

    # Per-map experiment dictionary
    general_experiment_dict = {}

    for map in maps:
        print("Gathering data from map: " + map)

        # Get the list of experiments
        experiments = os.listdir(experiments_path + map)
        experiments_set = (
            set()
        )  # Set with the name of experiments to warn duplicates in the same map
        general_experiment_dict[map] = {}  # Dict with the data of the experiments

        # For each experiment
        for experiment in experiments:
            # Get the experiment name from the config file
            config_file = open(
                experiments_path + map + "/" + experiment + "/config.txt", "r"
            )
            print("Gathering from experiment: " + experiment)
            experiment_name = ""
            data = {}
            for line in config_file:
                experiment_name += line.split(":")[1].strip() + "_"
                data[line.split(":")[0].strip()] = line.split(":")[1].strip()
            experiment_name = experiment_name[:-1]

            # Check if the experiment name is duplicated
            if experiment_name in experiments_set:
                print("Experiment name already exists!! Possible duplicated experiment")
                print("Experiment name: " + experiment_name)
            else:
                experiments_set.add(experiment_name)

            confusion_matrices = []
            for file in os.listdir(experiments_path + map + "/" + experiment):
                if ".txt" in file and "config" not in file:
                    step = int(file.split(".")[0])
                    file_path = experiments_path + map + "/" + experiment + "/" + file
                    confusion_matrix = np.loadtxt(file_path)
                    confusion_matrix = ConfusionMatrix(
                        step,
                        file_path,
                        confusion_matrix,
                    )
                    confusion_matrices.append(confusion_matrix)

            # Sort the confusion matrices by step
            confusion_matrices.sort(key=lambda x: x.step)

            current_experiment = ExperimentData(
                map,
                experiment_name,
                data["timestamp"],
                experiments_path + map + "/" + experiment,
                confusion_matrices,
                data,
            )

            # Append the experiment to the list
            general_experiment_dict[map][experiment_name] = current_experiment

    return general_experiment_dict


def get_experiments_names(
    experiment_data: Dict[str, Dict[str, ExperimentData]]
) -> List[str]:
    """
    Get the names of all the experiments, check for duplicates and completeness

    Args:
        experiment_data: Dictionary with the data of the experiments
    """
    experiments_names_set = set()

    for map_name in experiment_data:
        for experiment_name in experiment_data[map_name]:
            if experiment_name in experiments_names_set:
                print("Experiment name already exists!! Possible duplicated experiment")
                print("Experiment name: " + experiment_name)
            else:
                experiments_names_set.add(experiment_name)

    experiments_names = list(experiments_names_set)

    return experiments_names


def aggregate_results(experiment_data: Dict[str, Dict[str, ExperimentData]]) -> Dict[str, np.ndarray]:
    """
    Aggregate the results from the experiments

    Args:
        experiment_data: Dictionary with the data of the experiments
    """
    # Get the names of all the experiments
    experiments_names = get_experiments_names(experiment_data)
    num_classes = (
        experiment_data.values()[0].values()[0].confusion_matrices[-1].matrix.shape[0]
    )
    last_step = len(experiment_data.values()[0].values()[0].confusion_matrices)

    general_aggregated_confusion_matrix = {}

    # For each experiment
    for experiment_name in experiments_names:
        aggregated_confusion_matrix = np.zeros((num_classes, num_classes))
        # For each map
        for map_name in experiment_data:
            assert (
                len(experiment_data[map_name][experiment_name].confusion_matrices)
                == last_step
            ), "The number of steps is not the same for all the experiments"
            assert (
                experiment_data[map_name][experiment_name]
                .confusion_matrices[-1]
                .shape[0]
                == num_classes
            ), "The number of classes is not the same for all the experiments"
            # Get the confusion matrix of the experiment
            confusion_matrix = (
                experiment_data[map_name][experiment_name].confusion_matrices[-1].matrix
            )
            # Aggregate the confusion matrix
            aggregated_confusion_matrix += confusion_matrix

        # Save the aggregated confusion matrix
        general_aggregated_confusion_matrix[experiment_name] = aggregated_confusion_matrix

    return general_aggregated_confusion_matrix

def compute_metrics(matrix: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Compute the metrics from the confusion matrix

    Args:
        matrix: Confusion matrix

    Returns:
        Tuple with the metrics
    """
    # Compute the metrics
    accuracy = accuracy_score(matrix)
    precision = precision_score(matrix)
    recall = recall_score(matrix)
    f1 = f1_score(matrix)
    jaccard = jaccard_score(matrix)

    return accuracy, precision, recall, f1, jaccard

def main(experiments_path: Path, output_path: Optional[Path] = None):
    """
    Main script

    Args:
        experiments_path: Path to the experiments folder
        output_path: Path to the output folder
    """
    # If output path is not provided, use the experiments path
    if output_path is None:
        output_path = experiments_path / "results"

    # Create the output folder if it does not exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Gather the data from the experiments
    experiment_data = gather_experiment_data(experiments_path, output_path)

    # Aggregate the results
    aggregated_results = aggregate_results(experiment_data)

    # Compute the metrics
    metrics = {}
    for experiment_name in aggregated_results:
        metrics[experiment_name] = compute_metrics(aggregated_results[experiment_name])

    # Save the metrics in a csv file
    metrics_path = output_path / "metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("Experiment,Accuracy,Precision,Recall,F1,Jaccard\n")
        for experiment_name in metrics:
            f.write(
                f"{experiment_name},{metrics[experiment_name][0]},{metrics[experiment_name][1]},{metrics[experiment_name][2]},{metrics[experiment_name][3]},{metrics[experiment_name][4]}\n"
            )
            
    # Save the results
    aggregated_path = output_path / "aggregated"
    aggregated_path.mkdir(parents=True, exist_ok=True)
    for experiment_name in aggregated_results:
        np.savetxt(aggregated_path / f"{experiment_name}.txt", aggregated_results[experiment_name], fmt="%d")

    # Save the experiment data
    experiment_data_path = output_path / "experiment_data"
    experiment_data_path.mkdir(parents=True, exist_ok=True)
    for map_name in experiment_data:
        for experiment_name in experiment_data[map_name]:
            experiment_data[map_name][experiment_name].save(experiment_data_path)
