""" File I/O utilities. """
import csv
import glob
import os


def extract_config(configuration) -> (dict, dict, dict):
    """Configuration parameter extraction for simulation pipeline."""
    if configuration.get("synthesis", {}).get("method") == "sklearn":
        # point cloud data is synthesized with scikit-learn.
        synthesis_params = {
            # sklearn.make_classification keyword arguments, loaded directly from pipeline configuration file.
            "n_classes": configuration.get("synthesis", {}).get("sklearn", {}).get("classes"),
            "n_samples": configuration.get("synthesis", {}).get("sklearn", {}).get("samples"),
            "n_features": configuration.get("synthesis", {}).get("sklearn", {}).get("features"),
            "n_redundant": configuration.get("synthesis", {}).get("sklearn", {}).get("redundant"),
            "n_informative": configuration.get("synthesis", {}).get("sklearn", {}).get("informative"),
            "class_sep": configuration.get("synthesis", {}).get("sklearn", {}).get("separation"),
            "n_clusters_per_class": configuration.get("synthesis", {}).get("sklearn", {}).get("clusters_per_class"),
            "flip_y": configuration.get("synthesis", {}).get("sklearn", {}).get("flipped_label_prop"),
        }
    elif configuration.get("synthesis", {}).get("method") == "physical":
        # data with nontrivial, olfactory receptor-like geometry is synthesized directly.
        synthesis_params = {
            # sklearn.make_classification keyword arguments, loaded directly from pipeline configuration file.
            "seed_vector": configuration.get("synthesis", {}).get("physical", {}).get("seed_vector"),
            "n_samples": configuration.get("synthesis", {}).get("physical", {}).get("n_samples"),
            "depth": configuration.get("synthesis", {}).get("physical", {}).get("depth"),
            "sensors": configuration.get("synthesis", {}).get("physical", {}).get("sensors"),
            "sparsity": configuration.get("synthesis", {}).get("physical", {}).get("sparsity"),
            "signal": configuration.get("synthesis", {}).get("physical", {}).get("signal"),
            "contaminants": configuration.get("synthesis", {}).get("physical", {}).get("contaminants"),
            "noise": configuration.get("synthesis", {}).get("physical", {}).get("noise"),
            "sigmoid": configuration.get("synthesis", {}).get("physical", {}).get("sigmoid"),
        }
    else:
        synthesis_params = {}

    # for easier referencing of important configuration parameters throughout the script.
    sampling_params = {
        "repetitions": configuration.get("sampling", {}).get("repetitions", 0),
        "folds": configuration.get("sampling", {}).get("folds"),
        "shots": configuration.get("sampling", {}).get("shots", 1),
        "shuffle": configuration.get("sampling", {}).get("shuffle"),
        "batches": configuration.get("sampling", {}).get("included_batches", [1]),
        "train_batches": configuration.get("sampling", {}).get("train_batches", None),
        "test_batches": configuration.get("sampling", {}).get("test_batches", None),
        "key": configuration.get("sampling", {}).get("key", "Category"),
        "group_key": configuration.get("sampling", {}).get("group_key", ""),
    }
    simulation_params = {
        "dump": configuration.get("simulation", {}).get("dump"),
        "test": configuration.get("simulation", {}).get("test"),
        "noise": configuration.get("simulation", {}).get("noise", {}),
        "rsa": configuration.get("simulation", {}).get("rsa"),
        "svm": configuration.get("simulation", {}).get("svm"),
        "hetdup": configuration.get("simulation", {}).get("hetdup"),
        "render": configuration.get("simulation", {}).get("render"),
        "verbose": configuration.get("simulation", {}).get("verbose"),
        "tensorboard": configuration.get("simulation", {}).get("tensorboard"),
        "interactive": configuration.get("simulation", {}).get("interactive"),
        "train_duration": configuration.get("simulation", {}).get("train_duration"),
        "test_duration": configuration.get("simulation", {}).get("test_duration"),
        "rinse": configuration.get("simulation", {}).get("rinse"),
    }

    return synthesis_params, sampling_params, simulation_params


def get_files(path, extension, recursive=False):
    """A generator of filepaths for each file into path with the target extension."""
    if not recursive:
        for file_path in glob.iglob(path + "/*." + extension):
            yield file_path
    else:
        for root, dirs, files in os.walk(path):
            for file_path in glob.iglob(root + "/*." + extension):
                yield file_path


def to_csv(file: str, header: list, content: list):
    """Saves header and content to a CSV.

    Note
    ----
    Opens the file in append mode by default. Caller should ensure the correct file is added to.

    """
    with open(file + ".csv", "a") as f:
        writer = csv.writer(f, delimiter=",")

        if os.path.getsize(file + ".csv") == 0:
            writer.writerow(header)

        params = [str(p) for p in content]
        writer.writerow(params)
