""" Core EPL model. """
from typing import Any

import logging
import random

import numpy as np
from scipy import stats

import torch
from torch import Tensor

from alive_progress import alive_bar

from sapicore.engine.neuron.spiking import SpikingNeuron
from sapicore.engine.network import Network
from sapicore.model import Model

from sapicore.utils.constants import SEED

from src.models.component.synapse import IICSynapse
from src.utils.math import k_bits


__all__ = ("BulbModel", "BulbNetwork")


class BulbNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cfg = self.configuration.get("model", {})

        self.is_training = False
        self.genesis = cfg.get("neurogenesis")
        self.elapsed_cycles = 0

        # connect core EPL network with the random projection method or pick-k-gain-levels method.
        if "forward_prop" not in cfg:
            self.connect_epl(forward_connectivity_k=cfg.get("forward_k", 1))
        else:
            self.connect_epl(forward_connectivity_prop=cfg.get("forward_prop"))

        # connect signal conditioning layer.
        if kwargs.get("conditioning", True):
            # can be turned off by setting keyword argument `conditioning` to false at instantiation.
            self.connect_glom(enable_pg=cfg.get("ntco", False))

    def connect_epl(self, forward_connectivity_prop: float = None, forward_connectivity_k: int = 1):
        """Customizes connectivity of core EPL layers (MC, GC)."""
        # redefine the root layer to be MC (otherwise, the gamma oscillator will be auto-detected as root).
        self.roots = ["MC"]

        # easy reference to unit quantities.
        num_mc = self["MC"].num_units
        num_gc = self["GC"].num_units

        hd_factor = num_mc // self["ET"].num_units
        gc_mc_ratio = num_gc // num_mc

        if forward_connectivity_prop is not None:
            # MC->GC random connection probability.
            self["MC--GC"].connect("prop", forward_connectivity_prop)

        else:
            # for each sensor-specific subset of MCs, draw connections s.t. 'k' gain levels are represented from each
            # set of sensor-specific MCs. Orders of magnitude more efficient than random projection, and guaranteed to
            # include all valid MC->GC weight solution candidates.
            # uses helper function k_bits, which generates all binary strings of length `hd_factor` that have `k` 1s.
            temp = torch.zeros_like(self["MC--GC"].connections)

            # stores all possible draws for a slab of the appropriate dimensions.
            draw_list = k_bits(hd_factor, forward_connectivity_k)

            for gc in range(num_gc):
                # each slab's draw order should be different from the others.
                random.Random(SEED + gc).shuffle(draw_list)

                for slab in range(0, num_mc, hd_factor):
                    # set connections for this MC->GC slab.
                    temp[gc, slab : slab + hd_factor] = torch.as_tensor(
                        [int(x) for x in draw_list[(slab // hd_factor) % len(draw_list)]]
                    )

            self["MC--GC"].connections = temp

        # GC->MC connectivity is sparse.
        # NOTE in the fully connected post-neurogensis bulb network, GC[i % num_mc] -> MC[i % num_mc].
        self["GC--MC"].connections = torch.eye(num_mc).tile(gc_mc_ratio)[:, :num_gc].to(self.device)
        if self.genesis:
            enabled_gc = int(self.genesis.get("init") * num_gc)
            self["GC--MC"].connections[:, enabled_gc:] = 0

    def connect_glom(self, enable_pg: bool = False):
        """Customizes connectivity of the glomerular signal conditioning layer (ET, PG).

        Warning
        -------
        Connecting this layer will change the root node from MC to OSN.

        """
        # redefine the root layer to be the analog OSN layer.
        self.roots = ["OSN"]

        # easy reference to unit quantities.
        num_et = self["ET"].num_units
        num_mc = self["MC"].num_units

        # define connections for all signal conditioning layers except those that should be all-to-all (the default).
        self["OSN--ET"].connect("one")
        self["ET--ET"].connect("all")

        if enable_pg:
            num_pg = self["PG"].num_units

            self["OSN--PG"].connections = torch.eye(num_et).repeat_interleave(num_pg // num_et, 0).to(self.device)
            self["PG--ET"].connections = self["OSN--PG"].connections.T

            # since PG->ET is multiplicative, start at 1 to avoid zeroing out ET.
            self["PG"].voltage = torch.ones_like(self["PG"].voltage)

        # each ET projects to (num_mc // num_et) MCs, yielding a vertically expanded identity matrix shaped (MCs, ETs).
        self["ET--MC"].connections = torch.eye(num_et).repeat_interleave(num_mc // num_et, 0).to(self.device)

    def forward(self, data: torch.tensor) -> dict:
        if self.is_training and self.genesis:
            # enable a new slab of GCs at the requested rate.
            if self.simulation_step % self.genesis.get("freq") == 0:
                self.elapsed_cycles += 1

                num_mc = self["MC"].num_units
                num_gc = self["GC"].num_units

                gc_mc_ratio = num_gc // num_mc
                enabled_gc = int(
                    self.genesis.get("init") * num_gc + self.elapsed_cycles * self.genesis.get("rate") * num_gc
                )

                self["GC--MC"].connections = torch.eye(num_mc).tile(gc_mc_ratio)[:, :num_gc].to(self.device)
                self["GC--MC"].connections[:, enabled_gc:] = 0

        return super().forward(data)


class BulbModel(Model):
    """Core fitting, testing, and diagnostics. Reusable by derivative model classes."""

    def __init__(self, **kwargs):
        # instantiate the base bulb network or a derivative thereof from preexisting YAMLs.
        super().__init__(network=BulbNetwork(**kwargs))

        # dictionary for tracking and reporting convergence rate for each readout layer.
        self.converged = {}

    def calibrate(
        self,
        data: Tensor | list[Tensor],
        duration: int | list[int] | Tensor = 1,
        rinse: int | list[int] = 0,
        target_synapses: list[str] = None,
    ):
        """Wraps a model.fit() call that toggles off learning for all synapses except those listed in
        the `target_synapses` argument. After the model is calibrated, the original learning status is restored.

        Note
        ----
        For the bulb network, calibration is only intended to update ET->MC heterogeneous weights and
        should not affect weights anywhere else. Thus, call calibrate() with target_synapses=["ET--MC"].

        """
        # store original learning toggle status for each synapse.
        learning_status = {synapse.identifier: synapse.learning for synapse in self.network.get_synapses()}

        # toggle learning off for calibration except for target synapses.
        for synapse in self.network.get_synapses():
            self.network[synapse.identifier].set_learning(True if synapse.identifier in target_synapses else False)

        # calibrate the model, updating necessary synapses (in the bulb case, ET->MC, the quantization synapses).
        self.fit(data, duration, rinse, verbose=False, disable_learning_after_fit=False, is_training=False)

        # reset learning status to what it was for all synapses.
        for key, value in learning_status.items():
            self.network[key].set_learning(value)

        # reset refractory periods and voltages.
        for ens in self.network.get_ensembles():
            if isinstance(ens, SpikingNeuron):
                ens.refractory_steps = torch.zeros_like(ens.refractory_steps)
                ens.voltage = ens.volt_rest

        # reset last spiked logs.
        for syn in self.network.get_synapses():
            if isinstance(syn, IICSynapse):
                syn.src_last_spiked = torch.zeros_like(syn.src_last_spiked)
                syn.dst_last_spiked = torch.zeros_like(syn.dst_last_spiked)

    def fit(
        self,
        data: Tensor | list[Tensor],
        duration: int | list[int] | Tensor = 1,
        rinse: int | list[int] = 0,
        verbose: bool = True,
        disable_learning_after_fit: bool = True,
        weight_clamp_threshold: float = None,
        is_training=True,
    ):
        """Fits the model to a training set of `data` vectors, each presented for `duration` simulation steps.
        If `disable_learning_after_fit` is True, all synapses' learning switch will be turned off.

        Note
        ----
        Automatically toggles off learning when training is over. Turning advanced `diagnostics` on may
        increase runtime.

        """
        # flag that we are in training mode (e.g., to enable neurogenesis).
        self.network.is_training = is_training

        # wrap 2D tensor data in a list if need be, to accommodate both single- and multi-root cases.
        if not isinstance(data, list):
            data = [data]

        num_samples = data[0].shape[0]
        steps = (sum(rinse) if isinstance(rinse, list) else rinse) + (
            sum(duration) if isinstance(duration, list) else duration
        )

        with alive_bar(total=num_samples * steps, force_tty=True) as bar:
            # number of active and inactive synaptic connections, for tracking purposes.
            inactive = torch.count_nonzero(~self.network["MC--GC"].connections.bool())
            active = self.network["MC--GC"].weights.numel() - inactive

            # number of differentiated and pruned synaptic connections, for tracking purposes.
            diff = 0
            pruned = 0

            # automatically determine whether GC->MC weights or blocking durations should be logged.
            if isinstance(self.network["GC--MC"], IICSynapse):
                gc_mc_attr = "blocking_durations"
            else:
                gc_mc_attr = "weights"

            # log weight and blocking duration evolution for debugging purposes.
            mc_gc_evolution = torch.zeros(
                num_samples,
                self.network["MC--GC"].weights.shape[0],
                self.network["MC--GC"].weights.shape[1],
                device=self.network.device,
            )
            gc_mc_evolution = torch.zeros(
                num_samples,
                getattr(self.network["GC--MC"], gc_mc_attr).shape[0],
                getattr(self.network["GC--MC"], gc_mc_attr).shape[1],
                device=self.network.device,
            )

            # iterate over samples.
            for i in range(num_samples):
                # repeat each sample as many time steps as instructed (sample and hold).
                for k in range(duration if isinstance(duration, int) else duration[i]):
                    self.network.forward([data[j][i, :] for j in range(len(data))])
                    bar()

                # "rinse" network with null input if asked.
                for _ in range(rinse if isinstance(rinse, int) else rinse[i]):
                    self.network.forward([torch.zeros_like(data[j][i, :]) for j in range(len(data))])
                    bar()

                d_tmp = torch.count_nonzero(self.network["MC--GC"].weights.eq(self.network["MC--GC"].weight_max))
                p_tmp = torch.count_nonzero(self.network["MC--GC"].weights == 0) - inactive

                d_add = d_tmp - diff
                p_add = p_tmp - pruned

                diff = d_tmp
                pruned = p_tmp

                # if verbose:
                #    logging.warning(
                #        f"MC->GC differentiated {(diff / active) * 100:.2f}% (+{d_add}), "
                #        f"pruned {(pruned / active) * 100:.2f}% (+{p_add})"
                #    )

                mc_gc_evolution[i, :, :] = self.network["MC--GC"].weights
                gc_mc_evolution[i, :, :] = getattr(self.network["GC--MC"], gc_mc_attr)

        # if asked, learning is turned off after fitting the model to the training set.
        if disable_learning_after_fit:
            for synapse in self.network.get_synapses():
                synapse.set_learning(False)

        # if applicable, set MC->GC weights under a particular threshold to zero (e.g., undifferentiated weights).
        if weight_clamp_threshold is not None:
            mc_gc = self.network["MC--GC"]
            mc_gc.weights = mc_gc.weights * (mc_gc.weights > weight_clamp_threshold)

            # gc_mc = self.network["GC--MC"]
            # gc_mc.weights = gc_mc.weights * (gc_mc.weights < -weight_clamp_threshold)

        # turn off training flag.
        self.network.is_training = False

        return mc_gc_evolution, gc_mc_evolution

    def process(
        self,
        data: Tensor,
        repetitions: int | list[int] | Tensor = 1,
        rinse: int = 0,
        period: int = 50,
        conv_warn: int = 95,
    ) -> dict[str, Any]:
        """Passes a batch of samples to a network, returning population responses of the readout layers,
        having been subjected to a custom accumulation method (e.g., the method `sum` to get spike counts
        accumulated over each sample's exposure window).

        Parameters
        ----------
        data: Tensor
            Test samples to be fed to the trained network.

        repetitions: int or list of int or Tensor, optional
            Exposure duration for all/each sample, in simulation steps. Defaults to 1.

        rinse: int
            How long to present a 0s vector in-between samples. 0 by default to skip.

        period: int
            Background inhibitory oscillation period, in simulation steps. Defaults to 50.

        conv_warn: int
            Threshold below which to output a ConvergenceWarning, in terms of percentage of samples that
            failed to perfectly converge by the end of the testing period. Set to 95% by default.
            Warnings refer to a specific readout layer and are based on its spike output.

        Returns
        -------
        dict
            Dictionary whose keys are readout ensemble identifiers. Its values are 2D tensors logging the
            layer's responses to each sample, potentially after accumulation over test cycles.

        """
        # retrieve readout layer list from configuration dictionary.
        readout = self.network.configuration.get("processing", {}).get("readout", ["MC", "GC"])

        # initialize convergence dictionary to gauge performance of inhibitory plasticity rule.
        self.converged = {layer: [] for layer in readout}

        # wrap data in a list even if a single tensor was provided, to accommodate the multi-root network case.
        if not isinstance(data, list):
            data = [data]

        # placeholder for accumulated responses of each readout layer; one response tensor per sample.
        # lists are used because dimension of array returned from `accumulator` is unknown (sometimes vector, matrix).
        responses: dict[str, Any] = {layer: [] for layer in readout}
        steps = (sum(repetitions) if isinstance(repetitions, list) else repetitions) + rinse

        with (alive_bar(total=data[0].shape[0] * steps, force_tty=True) as bar):
            # iterate over samples.
            for i in range(data[0].shape[0]):
                # look up stimulus exposure duration if varies across samples.
                duration = repetitions if isinstance(repetitions, int) else repetitions[i]

                # buffer storing responses of each readout layer within a single exposure window.
                sample_responses = {
                    layer: torch.zeros((duration + rinse, self.network[layer].num_units), device=self.network.device)
                    for layer in readout
                }

                # allow network to process each test sample for as many time steps as instructed.
                for d in range(duration):
                    self.network([data[j][i, :] for j in range(len(data))])

                    for j, layer in enumerate(readout):
                        sample_responses[layer][d, :] = getattr(self.network[layer], self.network[layer].output_field)

                    bar()

                # "rinse" network with null input.
                for d in range(rinse):
                    self.network([torch.zeros_like(data[j][i, :]) for j in range(len(data))])

                    for j, layer in enumerate(readout):
                        sample_responses[layer][d + duration, :] = getattr(
                            self.network[layer], self.network[layer].output_field
                        )

                    bar()

                # only execute if convergence warning threshold > 0.
                # this is for verifying inhibitory learning rule correctness and incurs overhead.
                if conv_warn:
                    for layer in readout:
                        layer_resp = sample_responses[layer]
                        final_diff = sum(layer_resp[-rinse - period : -rinse, :]) - sum(
                            layer_resp[-rinse - 2 * period : -rinse - period, :]
                        )
                        final_diff = torch.as_tensor(torch.abs(final_diff) > 10**-5)

                        converged_prop = 100 - (torch.count_nonzero(final_diff) / layer_resp.shape[1]) * 100
                        self.converged[layer].append(converged_prop)

                # apply the accumulator function to each readout population's `sample_responses`.
                accumulators = self.network.configuration.get("processing", {}).get("accumulator", {})

                if accumulators:
                    for layer in readout:
                        try:
                            mode = accumulators.get(layer).get("mode")
                            take_final = accumulators.get(layer).get("take_final")
                        except KeyError as e:
                            print(f"Misconfigured accumulator: {e}")
                            responses[layer].append(sample_responses[layer])

                        responses[layer].append(
                            self.accumulator(
                                data=sample_responses[layer],
                                rinse=rinse,
                                period=period,
                                mode=mode,
                                take_final=take_final,
                            )
                        )
                else:
                    responses[layer].append(sample_responses[layer])

        # cast each accumulated response to a tensor.
        for key in responses.keys():
            responses[key] = torch.stack(responses[key])

        # print convergence summary if applicable.
        if conv_warn:
            for layer in readout:
                mean_conv = torch.mean(torch.as_tensor(self.converged[layer]))
                ci = stats.t.interval(
                    0.95,
                    len(self.converged[layer]) - 1,
                    loc=mean_conv,
                    scale=stats.sem(torch.as_tensor(self.converged[layer]).cpu()),
                )

                if mean_conv < conv_warn:
                    logging.warning(
                        f" {mean_conv:.3f}% of {layer} responses converged " f"(CI = {ci[0]:.2f} - {ci[1]:.2f}). "
                    )

        return responses

    @staticmethod
    def accumulator(data: Tensor, rinse: int, period: int = 25, mode: str = None, take_final: bool = True):
        match mode:
            case "sum":
                # NOTE plotting data.t() here would easily yield denoising figures by sample (Imam & Cleland, 2020).
                return sum(data[-rinse - period : -rinse, :]) if take_final else sum(data[:-rinse, :])

            case "mean":
                return torch.mean(data[-rinse - period : -rinse, :] if take_final else data[:-rinse, :], dim=0)

            case "phase":
                z = np.array(torch.argwhere(data.eq(1)).cpu())
                m = [np.where(z[:, 1] == i) for i in range(data.shape[1])]

                # compute mean preferred phase, since we need one value per neuron.
                preferred_phases = np.zeros(shape=data.shape[1])
                fired_cells = np.unique(z[:, 1])

                for cell in range(data.shape[1]):
                    if cell in fired_cells:
                        fired_phases = (z[m[cell]][:, 0] % period) + 1
                        if not take_final:
                            preferred_phases[cell] = np.mean(fired_phases) if fired_phases else 0
                        else:
                            preferred_phases[cell] = (
                                fired_phases[-1] if any(data[-rinse - period : -rinse, cell]) else 0
                            )

                return torch.as_tensor(preferred_phases)
