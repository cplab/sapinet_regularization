""" Default simulation pipeline, including similarity, discrimination, and utilization diagnostics. """
from typing import Type
from numpy.typing import NDArray

from copy import deepcopy
from datetime import datetime

import os
import gc
import random
import zipfile
import warnings

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import normalize

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns

from sapicore.pipeline import Pipeline
from sapicore.model import Model

from sapicore.data.sampling import CV, BalancedSampler

from sapicore.utils.constants import TIME_FORMAT
from sapicore.utils.io import ensure_dir, load_apply_config
from sapicore.utils.seed import fix_random_seed

from src.models.bulb import BulbModel

from src.utils.data import SNNData
from src.utils.data.io import to_csv, get_files, extract_config
from src.utils.data.loading import synthesize_data, explore_data
from src.utils.orchestration import MultiRun, parse_arguments

from src.utils.visual.plotting import (
    basic_plots,
    training_plots,
    line_plots,
    cell_recruitment,
    cell_responses,
    set_plot_theme,
)

from src.utils.ml.classification import SVM
from src.utils.ml.similarity import representational_analysis
from src.utils.ml.splitting import LimitedStratifiedKFold
from src.utils.math.distance import pairwise_wasserstein_distances

__all__ = ("Simulation",)


class Simulation(Pipeline):
    """SNN simulation pipeline with detailed diagnostic output.

    Note
    ----
    To reproduce Moyal et al. (2025):

        1. Run the current file as follows:
            python simulation.py -multirun yaml/hq2025.yaml

        2. Inspect the multirun output plots in `sapinet_regularization/results/<run>`.

        3. Open src/utils/analysis/analysis.r and change `run_dir` and `meta_dir` to the run output directory above.
        4. Run the R script and inspect the statistical results under sapinet_regularization/analysis/<run>.

    To further experiment with the simulator:

        1. Modify the configuration YAMLs at the project, model, or component level.
        2. Run the script with -experiment yaml/<your_configuration.yaml>.

    """

    def __init__(self, model_class: Type[Model], config: dict | str, data_dir: str, out_dir: str):
        """
        Parameters
        ----------
        model_class:
            Accepts any derivative of :class:`sapicore.model.Model`.
            The reference is used to create independent model instances during cross validation.

        config:
            Pipeline configuration dictionary or path to an equivalent YAML.

        data_dir:
            Directory containing processed external dataset(s), if applicable.

        out_dir:
            Directory to write results to. By default, the current timestamp is used to name the run.
        """
        # initialize generic pipeline attributes by calling the parent constructor.
        super().__init__(config_or_path=config)

        # session and condition names, used when assigning run output to subdirectories.
        self.session = self.configuration.get("session")
        self.condition = self.configuration.get("condition")
        self.variant = self.configuration.get("variant")
        self.mode = self.configuration.get("mode", "")

        # data and output directories associated with this pipeline instance.
        self.data_dir = data_dir
        self.out_dir = out_dir

        # dictionary for recording intermediate readout layer responses ({fold: {layer: response}}).
        # aggregated over the stimulus presentation duration (e.g., number of spikes fired).
        self.responses = {}
        self.readout_layers = []  # list storing references to readout layer objects once instantiated.
        self.fold = 0  # tracks cross validation fold for directory navigation purposes.

        # model class reference is specified to enable multiple instantiations (e.g., for different folds).
        self.model_class = model_class

        # whether to render plots or just save them to the output directory.
        self.render = self.configuration.get("simulation", {}).get("render", False)
        self.fig_size = self.configuration.get("fig_size", (25, 20))
        self.image_format = "svg"

        # default colormaps for different use cases.
        self.primary_cmap = self.configuration.get("primary_cmap", "viridis")  # all uses except below
        self.secondary_cmap = self.configuration.get("secondary_cmap", "rocket")  # filling matrices/heatmaps
        self.class_cmap = self.configuration.get("class_cmap", "Spectral")  # differentiating class labels

        # set random number generation seed across libraries for reproducibility.
        self.seed = fix_random_seed(self.configuration.get("seed", 314))

        # shorthand references to frequently used configuration fields, grouped into three dictionaries.
        self.syn_params, self.sam_params, self.sim_params = extract_config(self.configuration)

        # default matplot theme.
        set_plot_theme()

    def run(self):
        """Master script for generic simulation pipeline with diagnostics."""
        key = self.sam_params.get("key")  # name of metadata key whose values should be used as class labels.
        group_key = self.sam_params.get("group_key")  # name of metadata key whose values should be used for grouping.

        # generate or retrieve data based on the given configuration, then transfer to GPU if applicable.
        data = self._retrieve_data(self.mode, self.syn_params, self.sam_params, key)
        data.buffer.to(self.configuration.get("device"))

        # create output directories.
        ensure_dir(os.path.join(self.out_dir, "plots"))
        ensure_dir(os.path.join(self.out_dir, "tabular"))

        tr_ind = None
        te_ind = None

        # create stratified cross validation folds, potentially limiting training/testing to specific group(s).
        if self.sam_params["train_batches"] is not None:
            tr_ind = torch.as_tensor(data.metadata["batch"][:] == self.sam_params["train_batches"])
        if self.sam_params["test_batches"] is not None:
            te_ind = torch.as_tensor(data.metadata["batch"][:] == self.sam_params["test_batches"])

        cv = self._stratified_cv(
            data, self.sam_params["folds"], self.sam_params["shuffle"], key, tr_mask=tr_ind, te_mask=te_ind
        )

        # simulation logic in a cross validation loop ((test, train) per the few-shot nature of this simulation).
        for (i, (test, train)) in enumerate(cv):
            self._process_fold(data, train, test, self.sam_params, self.sim_params, key, group_key)

        # group-level plots based on responses accumulated across folds.
        if self.configuration.get("simulation", {}).get("group"):
            self.fold = False

            for layer in self.responses.keys():
                line_plots(self, self.responses[layer], self.responses[layer].metadata[key][:], log=False)
            line_plots(self, self.responses["Raw"], labels=self.responses["Raw"].metadata[key][:], log=True)

            for layer in ["ET", "MC"]:
                # NOTE group-level GC channel responses not included due to arbitrary mappings across folds.
                cell_responses(self, key, self.responses[layer], binary=False)

            for layer in self.readout_layers:
                is_phase = True if layer != "ET" else False  # MC/GC responses are phase-coded.
                is_util = True if layer != "ET" else False  # ET utilization is always 100%.
                is_dist = True  # whether to plot the spike phase distribution for each readout layer.

                # layer recruitment histogram (% active units).
                cell_recruitment(
                    self,
                    key,
                    self.responses[layer],
                    phase_mode=is_phase,
                    plot_utilization=is_util,
                    plot_distribution=is_dist,
                    dump_to_file=False,
                )

    def _process_fold(
        self, data: SNNData, train: NDArray, test: NDArray, samp_cfg: dict, sim_cfg: dict, key: str, group_key: str
    ):
        """Logic for a single cross validation fold.

        Parameters
        ----------
        data: SNNData
            Full dataset.

        train:
            Training set indices.

        test: NDArray
            Test set indices.

        samp_cfg: dict
            Configuration dictionary for sampling.

        sim_cfg: dict
            Configuration dictionary for simulation.

        key: str
            Metadata key for creating class labels.

        group_key: str
            Metadata key for grouping.

        """
        self.fold += 1

        # create fold-specific output subdirectory.
        ensure_dir(os.path.join(self.out_dir, "plots", f"F{self.fold}"))

        # prepare training and test samples.
        train_set, test_set = self._prepare_data(data, train, test, samp_cfg, sim_cfg, key)

        # load olfactory bulb configuration YAML from disk.
        bulb_path = os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "models", "yaml", "bulb", "bulb.yaml")
        )
        bulb_cfg = load_apply_config(bulb_path)

        # instantiate new network and model objects for this CV fold.
        model = self.model_class(configuration=bulb_cfg, device=self.configuration.get("device", "cpu"))
        print(f"Fold {self.fold}/{samp_cfg['folds']}: instantiating {model.network}")

        # store a zipped configuration snapshot, containing all YAML files in the run directory.
        snapshot = list(get_files(os.path.join(os.getcwd(), ".."), "yaml", recursive=True))
        with zipfile.ZipFile(os.path.join(self.out_dir, "config.zip"), "w") as zf:
            for file in snapshot:
                zf.write(file, compress_type=zipfile.ZIP_DEFLATED, arcname=os.path.basename(file))

        # reference to the rinse duration, as it is used frequently below.
        rinse = sim_cfg["rinse"]

        # attach data accumulator hooks for streaming intermediate results to disk, if applicable.
        if sim_cfg["dump"]:
            steps = (sim_cfg["train_duration"] + rinse) * train.shape[0]
            fold_directory = ensure_dir(os.path.join(self.out_dir, "data", f"F{self.fold}"))

            model.network.add_data_hook(data_dir=fold_directory, steps=steps)

        # adaptive calibration procedure, currently implemented only for BulbModel and its derivatives.
        if isinstance(model, BulbModel):
            # use training data to calibrate ET->MC quantization weights.
            print(f"Calibrating: {len(train)} samples.")

            # present each calibration sample for cycle length, to avoid mistiming later on.
            # remember that the entire network is active and keeping count during this step.
            cycle_length = int(1000 / model.network["OSC"].frequencies.item())
            model.calibrate(train_set[:], duration=cycle_length, rinse=50, target_synapses=["ET--MC"])

        # fit model to training data.
        print(
            f"Training: {len(train)} samples, {rinse}ms rinse + {sim_cfg['train_duration']}ms exposure."
            f"\nLabels: {train_set.metadata[key][:]}"
            + (f", Groups: {train_set.metadata[group_key][:]}." if key != group_key else ".")
        )
        weight_evo, blocking_evo = model.fit(
            train_set[:],
            rinse=rinse,
            duration=sim_cfg["train_duration"],
            verbose=sim_cfg["verbose"],
            weight_clamp_threshold=29,
        )

        # obtain readout layer list.
        self.readout_layers = model.network.configuration.get("processing", {}).get("readout", [])

        # save the trained model and its component diagram.
        ensure_dir(os.path.join(self.out_dir, "models"))
        model.save(os.path.join(self.out_dir, "models", f"F{self.fold}.pt"))

        # relevant independent variables, to streamline R analysis.
        duplication_factor = model.network["MC"].num_units // model.network["ET"].num_units
        resolution_threshold = model.network["ET--MC"].limit
        smoothing_factor = model.network["ET--MC"].smoothing

        # plot weight and blocking duration evolution.
        training_plots(
            self,
            weight_evo,
            train_set.metadata[key][:],
            x_range=[25, 30],
            title="MC->GC Histograms (Nonzero)",
            file_name="evolution_mc_gc",
        )

        # plot rudimentary post-training network diagnostics.
        basic_plots(self, model)

        if sim_cfg["test"]:
            # pass test samples through the trained network and obtain summarized readout layer responses.
            response_data = self._process_test(model, sim_cfg, test_set)

            # total MC/GC spikes across cycles, one line per sample, with x-axis being individual cells,
            # sorted most to least active, colored by class.
            for layer in response_data.keys():
                line_plots(self, response_data[layer], labels=test_set.metadata[key][:], log=False)
            line_plots(self, response_data["Raw"], labels=test_set.metadata[key][:], log=True)

            # raw cell responses, heatmap, preferred phase and binary.
            cell_responses(self, key, test_set, binary=False)

            for layer in self.readout_layers:
                cell_responses(self, key, response_data[layer], binary=False)

                # cell recruitment histograms, colored by class.
                is_phase = True if layer != "ET" else False
                is_util = False if layer == "ET" else True
                is_dist = True

                cell_recruitment(
                    self,
                    key,
                    response_data[layer],
                    phase_mode=is_phase,
                    plot_distribution=is_dist,
                    plot_utilization=is_util,
                    duplication_factor=duplication_factor,
                    resolution_threshold=resolution_threshold,
                    smoothing_factor=smoothing_factor,
                    shots=samp_cfg["shots"],
                    dump_to_file=True,
                )

            # visualize readout layer response manifolds.
            for layer in self.readout_layers:
                explore_data(
                    self,
                    response_data[layer],
                    key=key,
                    interactive=sim_cfg["interactive"],
                    path=os.path.join(
                        self.out_dir,
                        "plots",
                        f"F{self.fold}" if self.fold else "group",
                        f"projection_{layer}.{self.image_format}",
                    ),
                )

            # obtain representational similarity and deformation measures.
            if sim_cfg["rsa"]:
                representational_analysis(self, ["Raw"] + self.readout_layers, response_data, key)

            # estimate mapped manifold separability using leave-one-out CV and grid search on C and gamma.
            if sim_cfg["svm"]:
                for readout in self.readout_layers:
                    score, ci, confusion = self._separability_score(
                        response_data[readout],
                        key,
                        render=self.render,
                        duplication_factor=duplication_factor,
                        resolution_threshold=resolution_threshold,
                        smoothing_factor=smoothing_factor,
                    )
                    print(f"{readout} separability: {score:.2f} % (CI = {ci[0]:.2f} - {ci[1]:.2f})\n")

        # release memory.
        plt.close("all")
        torch.cuda.empty_cache()
        gc.collect()

        # fold separator for console output readability.
        print("------")

    def _process_test(self, model, sim_params, test_set):
        print(
            f"Testing: {test_set[:].shape[0]} samples, {sim_params['rinse']}ms rinse + {sim_params['test_duration']}ms exposure."
        )
        test_responses = model.process(test_set[:], sim_params["test_duration"], rinse=sim_params["rinse"])

        # initialize dictionary storing SNNData objects for readout layer responses.
        response_data = {}

        for layer in test_responses.keys():
            response_data[layer] = SNNData(
                identifier=f"{layer}", buffer=deepcopy(test_responses[layer]), metadata=deepcopy(test_set.metadata)
            )

        if not self.responses:
            # if this is the first CV iteration, initialize global response dataset.
            self.responses = deepcopy(response_data)

        else:
            # append current CV fold readout layer responses to global buffer for aggregation.
            for layer in response_data:
                self.responses[layer].concatenate(other=response_data[layer])

        # add raw vectors and their metadata to the global dataset as well.
        if "Raw" not in self.responses.keys():
            self.responses["Raw"] = SNNData(
                identifier="Raw", buffer=deepcopy(test_set[:]), metadata=deepcopy(test_set.metadata)
            )
        else:
            self.responses["Raw"].concatenate(test_set)

        # return raw inputs and readout layer responses from this fold for further processing.
        response_data["Raw"] = test_set

        return response_data

    def _prepare_data(self, data, train, test, samp_cfg, sim_cfg, key):
        # shuffle presentation order within train and test sets (the *indices* already selected for the two sets).
        if samp_cfg["shuffle"]:
            # avoid variations in order effects on MC->GC learning across runs, for reproducibility.
            train = train[random.sample(list(range(len(train))), len(train))]
            test = test[random.sample(list(range(len(test))), len(test))]

        # trim data object (buffer and metadata) to the specified train and test indices.
        train_set = data.trim(train)
        test_set = data.trim(test)

        # test set string identifier, for plotting purposes.
        test_set.identifier = "Raw"

        # test set noise settings (false if keys do not exist).
        noise_type = sim_cfg.get("noise", {}).get("mode")
        noise_add = sim_cfg.get("noise", {}).get("add")
        noise_inds = sim_cfg.get("noise", {}).get("inds")
        noise_args = sim_cfg.get("noise", {}).get("args")

        # "environmental" noise to add or replace test samples with, if applicable.
        if noise_type is not None:
            noise_ref = getattr(np.random, noise_type)
            test_set[:, noise_inds] = torch.as_tensor(
                noise_ref(size=test_set[:, noise_inds].shape, **noise_args),
                device=test_set[:].device,
                dtype=test_set[:].dtype,
            ) + (test_set[:, noise_inds] if noise_add else 0)

        # NOTE the data going into the model is raw; this deep copy is normalized for visualization purposes.
        temp = deepcopy(test_set)
        temp[:] = normalize(test_set[:], p=1)

        # visually explore synthesized test data if asked (otherwise, saves image to file from default viewpoint).
        if sim_cfg["interactive"]:
            explore_data(self, temp, key=key, interactive=True)

        return train_set, test_set

    def _retrieve_data(self, mode: str, synthesis_params: dict, sampling_params: dict, key: str):
        """Wrapper for data synthesis and/or sampling procedure.

        Note
        ----
        This preliminary version supports basic synthetic data generation or loading the UCSD drift set.
        In principle, these operations should be relegated to a separate module.

        """
        # generate synthetic dataset based on experiment configuration YAML.
        if mode == "Synthetic":
            print(f"Synthesizing data with the following properties: {synthesis_params}")
            data = synthesize_data(
                self,
                method=self.configuration.get("synthesis", {}).get("method"),
                params=synthesis_params,
                sampler=BalancedSampler(replace=True),
                duplicate=sampling_params["repetitions"] - 1,
                norm=False,
                key=key,
                group_keys=sampling_params["group_key"],
                shots=sampling_params["shots"],
                folds=sampling_params["folds"],
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return data

    def _stratified_cv(
        self,
        data: SNNData,
        folds: int,
        shuffle: bool,
        key: str | list[str],
        tr_mask: Tensor = None,
        te_mask: Tensor = None,
    ) -> CV:
        """Create stratified cross validation folds.

        When `mask` is not None, the indices specified by it should correspond to the group you wish to be smaller.
        In few-shot scenarios, mask indices should correspond to the training group and splitting should
        be `test, train in cv.split(...)`.

        """
        # sklearn raises ValueError if random_state is set when shuffle is False.
        if shuffle:
            cv = CV(
                data,
                LimitedStratifiedKFold(
                    n_splits=folds, tr_mask=tr_mask, te_mask=te_mask, shuffle=True, random_state=self.seed
                ),
                label_keys=key,
            )

        else:
            cv = CV(
                data,
                LimitedStratifiedKFold(n_splits=folds, tr_mask=tr_mask, te_mask=te_mask, shuffle=False),
                label_keys=key,
            )

        return cv

    def _separability_score(
        self,
        responses: SNNData,
        key: str,
        render: bool = True,
        duplication_factor: int = None,
        resolution_threshold: float = None,
        smoothing_factor: float = None,
    ):
        """Identifies optimal SVM hyperparameters using LOO cross validation on half the test set,
        then uses them in LOO CV on the other half to obtain a separability score (ease of classification
        of test points having been mapped to the manifold learned by Sapinet).

        In this basic version, test/validation is a single 2-split.

        Parameters
        ----------
        responses: SNNData
            Sapinet dataset whose buffer contains readout layer responses (to test samples) and whose
            metadata is their class labels.

        key:
            Metadata key (class name) of interest.

        render: bool
            Whether to render the confusion matrix as a seaborn heatmap plot.

        duplication_factor: int
            Network's duplication factor, defined as the number of MCs over the number of ETs.

        resolution_threshold: float
            Minimal value capable of eliciting MC spike response in one unit.

        smoothing_factor: float
            In adaptive weight density mode, smoothing factor applied to piecewise linear interpolation.

        Returns
        -------
        score: float
            Mean score over cross validation folds

        ci: list of float
            Confidence interval, 95% by default. Given as a list containing lower and upper bound.

        confusion: NDArray
            Confusion matrix.

        Note
        ----
        This measure is stable (not inflated) even when hyperparameter optimization and validation
        are both done on the entire test set; the 2-split is to preempt criticism.

        """
        print("Computing optimal SVM hyperparameters for response separability estimation.")

        # identifiers for logging.
        svm_header = {
            "Session": self.session,
            "Condition": self.condition,
            "Variant": self.variant,
            "Analysis": "SVM",
            "Mode": self.mode,
            "Layer": responses.identifier,
            "Duplication": duplication_factor,
            "Resolution": resolution_threshold,
            "Smoothing": smoothing_factor,
            "Batches": self.configuration.get("sampling", {}).get("included_batches", -1),
            "BatchesTr": self.configuration.get("sampling", {}).get("train_batches", -1),
            "BatchesTe": self.configuration.get("sampling", {}).get("test_batches", -1),
            "Shots": self.configuration.get("sampling", {}).get("shots", -1),
            "Fold": self.fold,
        }

        # split test set into validation set and remainder.
        validation, final_test = list(
            StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed).split(
                responses[:].cpu(), responses.metadata[key][:]
            )
        )[0]

        optimizer = SVM(
            identifiers=svm_header,
            data=responses[validation].cpu(),
            labels=responses.metadata[key][validation],
            cv=CV(
                data=responses.trim(validation),
                cross_validator=StratifiedKFold(
                    n_splits=self.configuration.get("sampling", {}).get("folds", 2) // 2,
                    shuffle=True,
                    random_state=self.seed,
                ),
                label_keys=key,
            ),
        )
        hyper, _ = optimizer.grid_svm(kernel="rbf", seed=self.seed)

        score, ci, confusion = SVM(
            identifiers=svm_header,
            data=responses[final_test].cpu(),
            labels=responses.metadata[key][final_test],
            cv=CV(
                data=responses.trim(index=final_test),
                cross_validator=StratifiedKFold(
                    n_splits=self.configuration.get("sampling", {}).get("folds", 2) // 2,
                    shuffle=True,
                    random_state=self.seed,
                ),
                label_keys=key,
            ),
        ).score_svm(
            kernel="rbf",
            c=hyper["C"],
            gamma=hyper["gamma"],
            flip=True,
            scale=(True if responses.identifier == "ET" else False),
            norm=False,
            seed=self.seed,
            file=os.path.join(self.out_dir, "tabular", f"sep_svm_{responses.identifier}"),
        )

        plt.figure(figsize=self.fig_size)
        u_labels = np.unique(responses.metadata[key][:])

        sns.heatmap(confusion, xticklabels=u_labels, yticklabels=u_labels, annot=True, fmt="g", rasterized=True)

        plt.title(f"{responses.identifier} confusion matrix")
        plt.xlabel("Predicted", fontsize=24)
        plt.ylabel("Actual", fontsize=24)

        plt.savefig(
            os.path.join(
                self.out_dir, "plots", f"F{self.fold}", f"confusion_{responses.identifier}.{self.image_format}"
            )
        )

        if render:
            plt.show()

        return score, ci, confusion

    def dump_utilization(
        self, df, duplication_factor, labels, resolution_threshold, responses, shots, smoothing_factor
    ):
        # pairwise earthmover distance between class-specific utilization histograms.
        # measures potential to confound systematic concentration differences with essential analyte properties.
        util_by_class = np.array([df[labels == k] for k in np.unique(labels)], dtype="object")
        em_distances = pairwise_wasserstein_distances(util_by_class)

        upper_triangular_indices = np.triu_indices(em_distances.shape[0], k=1)
        em_filtered = em_distances[upper_triangular_indices]

        emd_header = [
            "Session",
            "Condition",
            "Variant",
            "Analysis",
            "Mode",
            "Layer",
            "Duplication",
            "Resolution",
            "Smoothing",
            "Shots",
            "Fold",
            "Class1",
            "Class2",
            "EMD",
        ]

        # base directory for CSV output.
        out_base = os.path.join(self.out_dir, "tabular")

        for z in range(len(upper_triangular_indices[0])):
            emd_content = [
                self.session,
                self.condition,
                self.variant,
                "Utilization",
                self.mode,
                responses.identifier,
                duplication_factor,
                resolution_threshold,
                smoothing_factor,
                shots,
                self.fold,
                upper_triangular_indices[0][z],
                upper_triangular_indices[1][z],
                em_filtered[z],
            ]
            to_csv(os.path.join(out_base, f"emd_{responses.identifier}"), header=emd_header, content=emd_content)

        util_header = [
            "Session",
            "Condition",
            "Variant",
            "Analysis",
            "Mode",
            "Layer",
            "Duplication",
            "Resolution",
            "Smoothing",
            "Shots",
            "Fold",
            "Class",
            "Sample",
            "Utilization",
        ]

        for cls in range(util_by_class.shape[0]):
            for sample in range(len(util_by_class[cls])):
                util_content = [
                    self.session,
                    self.condition,
                    self.variant,
                    "Utilization",
                    self.mode,
                    responses.identifier,
                    duplication_factor,
                    resolution_threshold,
                    smoothing_factor,
                    shots,
                    self.fold,
                    cls + 1,
                    sample + 1,
                    util_by_class[cls][sample].item(),
                ]
                to_csv(os.path.join(out_base, f"util_{responses.identifier}"), header=util_header, content=util_content)


if __name__ == "__main__":
    args = parse_arguments()

    # timestamp for individual run directory.
    stamp = datetime.now().strftime(TIME_FORMAT)

    # create data and output directories if nonexistent.
    ensure_dir(args.data)
    ensure_dir(args.output)

    run_dir = ensure_dir(os.path.join(os.path.realpath(args.output), stamp)) if args.output else None
    dat_dir = ensure_dir(os.path.realpath(args.data)) if args.data else None

    # global plotting settings.
    plt.rcParams.update({"figure.max_open_warning": 0})

    # suppress irrelevant dependency-invoked warnings encountered during development.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not args.multirun:
        # run a single simulation from the given pipeline configuration dictionary.
        Simulation(model_class=BulbModel, config=args.experiment, data_dir=dat_dir, out_dir=run_dir).run()
    else:
        # run multiple simulations from the given multi-run and pipeline configuration dictionaries.
        MultiRun(
            config=args.multirun,
            out_dir=run_dir,
            child_kwargs={"model_class": BulbModel, "config": args.experiment, "data_dir": dat_dir, "out_dir": run_dir},
        ).run()
