""" Plotting utility functions. """
import os
import numpy as np

import torch
from torch import Tensor

import matplotlib
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sapicore.utils.constants import SYNAPSE_SPLITTERS

from src.utils.data import SNNData
from src.utils.math import map_words_to_integers


def attribute_plots(
    pipeline, model, attributes: dict[str, str | list[str]], active_only: bool = True, plot_type: str = "hist", **kwargs
):
    """Plots select model component attributes.

    Parameters
    ----------
    pipeline: Simulation
        Configured simulation pipeline instance.

    model:
        Accepts all Sapicore Model derivatives.

    attributes: dict
        Dictionary specifying which attributes should be plotted for each named component (key).
        Formatted, e.g., {"MC->GC": ["weight"], "GC->MC": ["blocking_duration"]}.

    active_only: bool
        When true, if component is a synapse, only includes elements marked active in the `connections` matrix.

    plot_type: str
        Only "hist" and "heatmap" currently accepted.

    Note
    ----
    This method and others like it will eventually be relegated to utils.visual.

    """
    for comp, attributes in attributes.items():
        if not isinstance(attributes, list):
            # cast if user supplied a single attribute not wrapped in a list.
            attributes = [attributes]

        for attr in attributes:
            # retrieve current attribute from component.
            elements = getattr(model.network[comp], attr)

            # if component is a synapse and asked to filter inactive connections.
            if any([s in comp for s in SYNAPSE_SPLITTERS]) and active_only:
                elements = elements[torch.where(model.network[comp].connections == 1)]

            match plot_type:
                case "hist":
                    df = elements.flatten().cpu()
                    plt.figure(figsize=pipeline.fig_size)
                    sns.histplot(data=df, rasterized=True)

                case "heatmap":
                    d = elements if not active_only else elements.reshape(1, elements.shape[0])
                    d = d[np.argsort(kwargs["labels"])] if "labels" in kwargs else d

                    plt.figure(figsize=pipeline.fig_size)
                    sns.heatmap(
                        d.cpu(),
                        cmap=pipeline.primary_cmap,
                        rasterized=True,
                        norm=LogNorm() if kwargs.get("log_norm") else None,
                    )
                    plt.grid(False)

                case _:
                    raise RuntimeError(f"Unknown plot type requested ({plot_type}).")

            plt.title(f"{comp} {attr.replace('_', ' ')}")
            plt.savefig(
                os.path.join(
                    pipeline.out_dir,
                    "plots",
                    f"F{pipeline.fold}" if pipeline.fold else "group",
                    f"{attr}_{comp}_{plot_type}.{pipeline.image_format}",
                )
            )

            if pipeline.render:
                plt.show()

            plt.clf()
            plt.close("all")


def basic_plots(pipeline, model):
    # weight and blocking duration histograms.
    attribute_plots(
        pipeline,
        model,
        attributes={"MC->GC": "weights", "GC->MC": "blocking_durations"},
        active_only=True,
        plot_type="hist",
    )
    # weight heatmaps.
    attribute_plots(
        pipeline,
        model,
        attributes={"ET->MC": "weights", "MC->GC": "weights", "GC->MC": "blocking_durations"},
        active_only=False,
        plot_type="heatmap",
        log_norm=False,
    )

    # GC order plot.
    order_plots(pipeline, synapse=model.network["MC->GC"], threshold=25)


def line_plots(pipeline, responses, labels, sort=True, log=False):
    """Class-colored sorted line plots depicting population responses as accumulated."""
    sorted_responses = np.flip(np.sort(responses[:].cpu()), axis=1) if sort else np.array(responses[:].cpu())

    if not pipeline.render:
        matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=pipeline.fig_size)
    ax.plot(sorted_responses.transpose(), "-", label=labels, alpha=1.0, linewidth=1.0)

    if log:
        ax.set_yscale("log")

    colormap = plt.get_cmap(pipeline.class_cmap)
    colors = [colormap(i) for i in np.linspace(0, 1, len(np.unique(labels)))]

    for y, z in enumerate(ax.lines):
        z.set_color(colors[map_words_to_integers(labels)[y] - 1])

    plt.title(f"Sorted {responses.identifier}")
    plt.savefig(
        os.path.join(
            pipeline.out_dir,
            "plots" + (os.sep + f"F{pipeline.fold}" if pipeline.fold else ""),
            f"sorted_line_{responses.identifier}{'_log' if log else ''}.{pipeline.image_format}",
        )
    )

    if pipeline.render:
        plt.show()

    plt.clf()
    plt.close("all")


def cell_recruitment(
    pipeline,
    key: str,
    responses: SNNData,
    plot_utilization: bool = True,
    plot_distribution: bool = True,
    phase_mode: bool = True,
    duplication_factor: int = None,
    resolution_threshold: float = None,
    smoothing_factor: float = None,
    shots: int = -1,
    dump_to_file: bool = True,
):
    """Computes proportion of readout layer cells (whose responses were passed to this method)
    that have fired and the distribution of their responses.

    Optionally:
        Renders/saves histograms, colored by class and stacked.
        Dumps relevant utilization data to CSVs, for further processing.

    Note
    ----
    This manner of plotting emphasizes variability, however small.

    To better communicate utilization% uniformity (where it exists), use:
    plt.bar(height=torch.count_nonzero(responses[sorted_indices], dim=1), x=np.arange(responses[:].shape[0]))

    """
    sorted_indices = np.argsort(responses.metadata[key][:])
    labels = responses.metadata[key][sorted_indices]

    colors = [plt.get_cmap(pipeline.class_cmap)(i) for i in np.linspace(0, 1, len(np.unique(labels)))]
    df = (torch.count_nonzero(responses[sorted_indices], dim=1) / responses[:].shape[1] * 100.0).reshape(
        responses[:].shape[0], 1
    )

    if plot_utilization:
        # process and save utilization data for later R analysis.
        if dump_to_file:
            pipeline.dump_utilization(
                df.cpu(),
                duplication_factor,
                labels,
                resolution_threshold,
                responses,
                shots,
                smoothing_factor,
            )

        plt.figure(figsize=pipeline.fig_size)
        plot_histogram(
            pipeline,
            "Cell Recruitment (%)",
            "recruitment",
            responses.identifier,
            colors,
            df.cpu(),
            labels,
            bins=50,
            range=(0, 100),
        )
        if pipeline.render:
            plt.show()
        plt.clf()

    if plot_distribution:
        plt.figure(figsize=pipeline.fig_size)
        plot_histogram(
            pipeline,
            "Distribution",
            "dist",
            responses.identifier,
            colors,
            responses[sorted_indices].cpu(),
            labels,
            bins=50,
            range=(0, 50) if phase_mode else None,
        )
        if pipeline.render:
            plt.show()
        plt.clf()

    plt.close("all")


def training_plots(
    pipeline,
    evolution: Tensor,
    labels: list,
    x_range: list,
    nonzero: bool = True,
    title: str = "",
    file_name: str = "",
):
    """Helper function for training diagnostics."""
    num_samples = len(labels)
    num_classes = len(np.unique(labels))
    spc = num_samples // num_classes
    word_ids = map_words_to_integers(labels)

    colormap = plt.get_cmap(pipeline.primary_cmap)
    colors = [colormap(i) for i in np.linspace(0, 1, num_classes)]

    # create shot X analyte panel.
    fig = plt.figure(figsize=pipeline.fig_size, dpi=600)
    fig.suptitle(title)

    panels = fig.subfigures(nrows=spc, ncols=1)
    if spc == 1:
        panels = [panels]

    for row, sub in enumerate(panels):
        axes = sub.subplots(nrows=1, ncols=num_classes)
        for col, ax in enumerate(axes):
            sample = evolution[row * num_classes + col]
            analyte = word_ids[row * num_classes + col]
            rng = x_range[1] - x_range[0]

            ax.set_title(f"Sample {(row * num_classes + col) + 1}\nAnalyte {analyte}")
            ax.hist(
                sample[sample != 0].cpu() if nonzero else sample.flatten().cpu(),
                color=colors[analyte - 1],
                bins=50 if rng >= 50 else rng,
                range=(x_range[0], x_range[1]),
                density=True,
            )
            ax.set_ylim(0, 1)

    if file_name:
        plt.savefig(
            os.path.join(
                pipeline.out_dir,
                "plots" + (os.sep + f"F{pipeline.fold}" if pipeline.fold else ""),
                f"{file_name}.{pipeline.image_format}",
            )
        )

    if pipeline.render:
        plt.show()
    plt.clf()
    plt.close("all")


def plot_histogram(
    pipeline, title: str, prefix: str, identifier: str, colors: list, df: Tensor, labels: Tensor | list, **kwargs
):
    # n_classes = len(np.unique(labels))
    # if df.shape[1] == 1:
    #    df = df.reshape(df.shape[0]//n_classes, n_classes)

    plt.clf()
    plt.hist(
        df.T.cpu(),
        histtype="bar",
        color=[colors[int(i) - 1] for i in map_words_to_integers(labels)],
        stacked=True,
        fill=True,
        rasterized=True,
        **kwargs,
    )
    plt.title(f"{identifier} {title}")
    plt.savefig(
        os.path.join(
            pipeline.out_dir,
            "plots" + (os.sep + f"F{pipeline.fold}" if pipeline.fold else ""),
            f"{prefix}_{identifier}.{pipeline.image_format}",
        )
    )
    plt.close("all")


def cell_responses(pipeline, key: str, responses: SNNData, binary: bool = False):
    """Plots raw cell responses as a heatmap."""
    sorted_indices = np.argsort(responses.metadata[key][:])

    if binary:
        df = responses[sorted_indices].bool().int().cpu()
    else:
        df = responses[sorted_indices].cpu()

    plt.clf()
    plt.figure(figsize=pipeline.fig_size)
    sns.heatmap(df, cmap="rocket", rasterized=True)
    plt.title(f"{responses.identifier}, sorted by class")
    plt.savefig(
        os.path.join(
            pipeline.out_dir,
            "plots" + (os.sep + f"F{pipeline.fold}" if pipeline.fold else ""),
            f"responses_{responses.identifier}.{pipeline.image_format}",
        )
    )

    if pipeline.render:
        plt.show()
    plt.close("all")


def order_plots(pipeline, synapse, threshold: float = 0.0):
    """Compute and plot postsynaptic cell orders, defined by number of incoming weights > `threshold`
    (between 0 and 1). The threshold is expressed as a percentile.

    """
    plt.clf()
    plt.figure(figsize=pipeline.fig_size)

    orders = torch.count_nonzero(synapse.weights * (synapse.weights > threshold), dim=1)
    sns.histplot(orders.cpu(), bins=len(torch.unique(orders)), rasterized=True)

    plt.title(f"Cell orders (number of differentiated {synapse.identifier} weights > {threshold})")
    plt.savefig(
        os.path.join(
            pipeline.out_dir,
            "plots",
            f"F{pipeline.fold}",
            f"weights_th_{threshold}_{synapse.identifier}.{pipeline.image_format}",
        )
    )

    if pipeline.render:
        plt.show()
    plt.close("all")


def set_plot_theme(width: float = 12, height: float = 6, font: float = 15):
    """Universal plot theme reproducing Moyal et al. (2025)."""

    plt.rcParams["figure.figsize"] = (width, height)
    plt.rcParams["font.size"] = font

    # plt.rcParams['font.family'] = 'T'
    plt.rcParams["axes.labelsize"] = plt.rcParams["font.size"]
    plt.rcParams["axes.titlesize"] = 1.5 * plt.rcParams["font.size"]
    plt.rcParams["legend.fontsize"] = plt.rcParams["font.size"]
    plt.rcParams["xtick.labelsize"] = plt.rcParams["font.size"]
    plt.rcParams["ytick.labelsize"] = plt.rcParams["font.size"]

    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.loc"] = "center left"
    plt.rcParams["axes.linewidth"] = 3.0

    plt.gca().spines["right"].set_color("none")
    plt.gca().spines["top"].set_color("none")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("left")
