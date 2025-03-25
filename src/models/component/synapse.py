""" Custom synapse models. """
import torch

from torch import Tensor
from torch.nn.functional import normalize

from sapicore.engine.synapse import Synapse
from sapicore.engine.synapse.STDP import STDPSynapse

from src.utils.math import tensor_linspace, density_from_breaks

__all__ = ("DecaySTDP", "IICSynapse", "QuantizationSynapse", "ShuntSynapse")

W_EPSILON = 0.001  # threshold for wiping out IIC synapse weights that approach 0.
K = []
L = []
M = []
N = []


class QuantizationSynapse(Synapse):
    """Controls the mapping from an analog source layer to a spiking destination layer.

    Uniform mode: sets quantization weights to be either 1/x or equidistant in the range [`self.limit`, 1].
    The former encodes a uniform prior over the input space; the latter, an exponential prior.

    Adaptive mode: learns the distribution of source activity values over multiple presentations.
    Weights are set piecewise, as a function of a smoothing parameter controlling the grain of the approximation.
    Within each segment, based on user preference, weights can be either 1/x or linearly spaced between
    running extremes identified independently for each feature (e.g., different sensor may have different
    unknown response distributions).

    """

    def __init__(self, **kwargs):
        # base synapse initialization.
        super().__init__(**kwargs)

        # number of initial features (e.g., number of OSNs/ETs).
        self.num_features = self.src_ensemble.num_units

        # infers heterogeneous duplication factor based on number of units in the target layer.
        self.duplication_factor = self.dst_ensemble.num_units // self.num_features

        # analog value that would activate one spiking neuron in the target column.
        self.limit = self.configuration.get("model").get("limit", 1e-3)

        # current required to fire a spiking neuron in the target column, as parameterized.
        self.payload = self.configuration.get("model").get("payload", 1.0)

        # whether homogeneous duplication should be used as a baseline.
        self.baseline = self.configuration.get("model").get("baseline", False)

        # whether the naive prior in heterogeneous mode is uniform (1/x weights) or geometric (linspace weights).
        self.uniform = self.configuration.get("model").get("uniform", True)

        # adaptive weight density mode.
        self.adaptive = self.configuration.get("model").get("adaptive", False)

        # smoothing factor controlling bias-variance with adaptive calibration; no smoothing by default.
        self.smoothing = self.configuration.get("model").get("smoothing", 0)

        # store piecewise approximation space pre adjustment for LIF parameterization.
        self.space = None

        # tensors for storing running minimums and maximums for each feature.
        # start at +-maximal representable float to avoid exceptions on +-inf for flat priors.
        self.running_min = (torch.finfo(torch.float).max * torch.ones(self.num_features)).to(self.device)
        self.running_max = -torch.finfo(torch.float).max * torch.ones(self.num_features).to(self.device)

        # log calibration observations for adaptive density mechanism.
        self.observations = torch.zeros(self.num_features, device=self.device)

    def update_weights(self) -> Tensor:
        """Updates quantization weights during calibration procedure."""
        # initialize weight space.
        space = torch.zeros((self.duplication_factor, len(self.running_min)), device=self.device)

        # due to normalization, the number of sensors dictates where "flat" distributions would end up.
        # thus, we can target an expected median utilization of ~50% by setting a gain factor to sensors*payload.
        # the above applies in the uniform, non-adaptive mode (not informed by data).
        # in adaptive mode, weights naturally skew to where the data is, so gain correction isn't necessary.
        gain = self.payload

        if self.baseline:
            # homogeneous duplication mode; all weights are the same.
            space = space + self.baseline

            # in baseline mode, all weights are the same, so gain correction isn't necessary.
            gain = 1

        elif not self.adaptive:
            # in the non-adaptive case, the value space to be covered by the weights is always [self.limit, 1].
            if self.uniform:
                # 1/x weights, corresponding to a uniform prior on the input distribution.
                value_space = tensor_linspace(
                    torch.ones_like(self.running_max),
                    torch.zeros_like(self.running_max) + (1 / self.limit),
                    self.duplication_factor,
                )
                space = (1 / self.limit) / value_space

            else:
                # equidistant weights, corresponding to an exponential prior on the input distribution.
                space = tensor_linspace(
                    1 / torch.ones_like(self.running_max),
                    1 / (torch.zeros_like(self.running_min) + self.limit),
                    self.duplication_factor + 1,
                    device=self.device,
                )[:-1]

        else:
            # generate weights with adaptive densities, corresponding to a nonuniform prior.
            # here, sensor running max values bound the range covered by the weights, [self.limit, max_i].
            for i in range(space.shape[1]):
                # weight spacing during interpolation can be 1/x or linear, determined by `self.uniform`.
                space[:, i] = torch.as_tensor(
                    density_from_breaks(
                        self.observations[:, i],
                        factor=self.duplication_factor,
                        smoothing=self.smoothing,
                        inf_replacement=self.limit,
                        uniform=self.uniform,
                    ),
                    device=self.device,
                )

        # sort the results ascending.
        space = torch.sort(space, dim=0)[0]

        # adaptive procedure changes weight "center of mass"; gain must be corrected to center the distribution.
        if self.adaptive:
            correction = torch.median(torch.linspace(self.limit, 1, self.duplication_factor)) / torch.median(1 / space)
            gain = gain / correction

        # self.plot_approximation(space, self.observations)

        # update the weights, ensuring they stay on the original hardware device.
        self.weights = 0 * self.weights
        n = self.duplication_factor

        for i in range(space.shape[1]):
            self.weights[n * i : n * (i + 1), i] = gain * torch.as_tensor(space[:, i], device=self.weights.device)

        return self.weights

    @staticmethod
    def plot_approximation(space: Tensor, observations: Tensor):
        """Plot piecewise approximation next to sorted observation curves for illustration purposes."""
        import matplotlib.pyplot as plt

        plt.clf()
        limits = (
            torch.min(torch.min(1 / space), torch.min(observations)),
            torch.max(torch.max(1 / space), torch.max(observations)),
        )

        plt.figure(figsize=(25, 20))
        plt.title(f"Values Covered by Adaptive Weights (Step {observations.shape[0]-1})")
        [plt.plot(torch.sort(1 / space[:, k]).values.__reversed__()) for k in range(space.shape[1])]
        plt.ylim(limits)
        plt.show()

        plt.figure(figsize=(25, 20))
        plt.title(f"Calibration Observations (Step {observations.shape[0]-1})")
        [plt.plot(torch.sort(observations[1:, k]).values.__reversed__()) for k in range(observations.shape[1])]
        plt.ylim(limits)
        plt.show()

        plt.close("all")

    def forward(self, data: Tensor) -> dict:
        if self.learning:
            # log observation for adaptive calibration.
            self.observations = torch.vstack((self.observations, torch.as_tensor(data, device=self.device)))

            # data is a 1D sample vector. 0s should be disregarded (filtered).
            new_minima = (data < self.running_min) * (data != 0)
            new_maxima = (data > self.running_max) * (data != 0)

            # only update weights when necessary.
            if any(new_maxima) or any(new_minima):
                self.running_min[new_minima] = data[new_minima]
                self.running_max[new_maxima] = data[new_maxima]

                self.update_weights()

        return super().forward(data)


class DecaySTDP(STDPSynapse):
    """STDP synapse with a constant weight decay applied on every N-th step.

    Parameters
    ----------
    decay: Tensor
        Matrix specifying the amount to subtract from each weight on relevant simulation steps.

    interval: Tensor
        Decay will be applied on every simulation step divisible by this number (e.g., on every 5th step).
        By default, applies decay on every step (1). Matrix format supported for specifying variable intervals.

    non_causation_penalty: float
        Value by which to decrease weights where target spiked but source did not (within the same cycle).

    """

    _config_props_: tuple[str] = STDPSynapse._config_props_ + ("decay", "interval", "non_causation_penalty")

    def __init__(
        self,
        decay: Tensor = None,
        interval: Tensor = None,
        non_causation_penalty: float = 50,
        cycle_length: int = 50,
        **kwargs,
    ):

        # default STDP synapse instantiation.
        super().__init__(**kwargs)

        self.decay = decay if decay else torch.zeros_like(self.weights)
        self.interval = interval if interval else torch.ones_like(self.weights)

        self.cycle_length = cycle_length

        # optional decay value for synapses where target spiked but source did not.
        self.non_causation_penalty = self.configuration.get("model").get("non_causation_penalty", 0)

    def update_weights(self) -> Tensor:
        """Overrides weight update implementation with hSTDP rules specified in Imam & Cleland (2020).

        Returns
        -------
        Tensor
            2D tensor containing weight deltas for this forward pass (dW).

        """
        # update last spike time to current simulation step if a spike has just arrived at the synapse.
        # NOTE a delayed view of the presynaptic spike status is used here, simulating correct "arrival" time.
        self.src_last_spiked = self.src_last_spiked + self.delayed_data * (-self.src_last_spiked + self.simulation_step)

        # update last spike time only for postsynaptic units that spiked in this model iteration.
        self.dst_last_spiked = self.dst_last_spiked + self.dst_ensemble.spiked * (
            -self.dst_last_spiked + self.simulation_step
        )

        # synapse element pairs that were maxed out or where postsynaptic element never spiked should not be updated.
        inclusion = torch.ones_like(self.weights, dtype=torch.int)

        inclusion[self.weights == self.weight_max] = 0
        inclusion[:, self.src_last_spiked == 0] = 0
        inclusion[self.dst_last_spiked == 0, :] = 0

        # initialize dW matrix, formatted dst.num_units X src.num_units.
        delta_weight = torch.zeros(self.matrix_shape, dtype=torch.float, device=self.device)

        # skip computation if inclusion tensor is all 0s (e.g., during rinse periods).
        if torch.count_nonzero(inclusion) == 0:
            return delta_weight

        # subtract postsynaptic last spike time stamps from presynaptic:
        # transpose dst_last_spiked to a column vector, repeat column-wise, then add its negative to src_last_spiked.
        dst_tr = self.dst_last_spiked.reshape(self.dst_last_spiked.shape[0], 1)

        # NOTE since we subtract src-dst, negative values warrant LTP.
        delta_spike = -dst_tr.repeat(dst_tr.shape[1], self.src_ensemble.num_units) + self.src_last_spiked

        # spike time differences for the potentiation and depression cases (pre < post, pre > post, respectively).
        ltp_diffs = (delta_spike < 0.0) * delta_spike
        ltd_diffs = (delta_spike > 0.0) * delta_spike

        # add to total delta weight matrix (diffs are in simulation steps, tau are in ms).
        delta_weight = delta_weight + (delta_spike < 0.0) * (
            self.alpha_plus * torch.exp(ltp_diffs / (self.tau_plus / self.dt))
        )
        delta_weight = delta_weight + (delta_spike > 0.0) * (
            -self.alpha_minus * torch.exp(-ltd_diffs / (self.tau_minus / self.dt))
        )

        # only change weights that made it to the inclusion list.
        updated_weights = self.weights.add(inclusion * delta_weight) * self.connections

        # weights never truly reach zero, so we need to intentionally zero out < epsilon (defined above).
        pruned_connections = torch.abs(self.weights) < W_EPSILON * torch.abs(self.weight_max)

        # if postsynaptic spiked but presynaptic didn't, connection should decay (unless differentiated).
        penalty = ((delta_spike < -self.cycle_length) * self.non_causation_penalty) * inclusion

        # set weights to updated values, ensuring pruned connections are zeroed out in the process.
        self.weights = (updated_weights - penalty) * ~pruned_connections + 0.0 * pruned_connections

        return delta_weight

    def forward(self, data: Tensor) -> dict:
        # apply specified weight decay(s) to cells in accordance with their individual intervals.
        # fully differentiated synapses (at w_max) are exempt from decay.
        if self.learning:
            # decay intervals may vary across synapse elements, so this step must be performed.
            is_decay = self.simulation_step % self.interval == 0
            self.weights = self.weights - is_decay * (self.weights < self.weight_max) * self.decay

        # normal STDP synapse operations, including a call to the overridden update_weights.
        result = super().forward(data)

        return result


class L1Synapse(Synapse):
    """L1 normalization synapse meant to connect an analog neuron layer to itself.

    This synapse directly divides the voltage vector of the source ensemble by its sum.

    Warning
    -------
    Only usable for self-connections from an ensemble to itself. Efficient but quite un-neuromorphic.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, data: Tensor) -> dict:
        super().forward(data)

        # no matter what, the output of this layer should be normalized to the range [0, 1].
        self.dst_ensemble.voltage = normalize(
            self.dst_ensemble.voltage.reshape(1, self.dst_ensemble.voltage.shape[0]), p=1.0
        ).flatten()

        return self.loggable_state()


class ShuntSynapse(Synapse):
    """Shunting inhibition with instant, same-cycle action on the destination ensemble voltage.

    This synapse computes the elementwise inverse of the default synapse output,
    with inf values replaced by 1 (identity), then multiplies the destination ensemble's voltage by it.

    Warnings
    --------
    Designed to project to an ensemble with multiplicative aggregation.

    Instant action is only necessary where lateral shunting inhibition must act before the signal can
    propagate to deeper layers. When in doubt, favor a dual compartment design over instant lateral action.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.output = torch.ones_like(self.output)

    def forward(self, data: Tensor) -> dict:
        # standard synapse forward operations.
        super().forward(data)

        # emit the inverted sum of incoming conductances (1 / W*x).
        self._invert_output()

        # instantly shunt the destination ensemble's activity.
        self.dst_ensemble.voltage = self.dst_ensemble.voltage * self.output

        return self.loggable_state()

    def _invert_output(self):
        # invert the output (1 / W*x), division by zero handled by converting inf to 1.
        self.output = torch.ones_like(self.output) / self.output
        self.output[self.output == float("inf")] = 1.0


class MultSynapse(Synapse):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, data: Tensor) -> dict:
        # use a list of queues to track transmission delays and "release" input after the fact.
        self.delayed_data = self.queue_input(data)

        # enforce weight limits.
        self.weights = torch.clamp(self.weights, self.weight_min, self.weight_max)

        # mask non-connections (repeated on every iteration in case mask was updated during simulation).
        self.weights = self.weights.multiply_(self.connections)

        def _product_aggregation(w: Tensor, d: Tensor) -> Tensor:
            """Multiplies products of nonzero weights and corresponding data values."""
            result = torch.zeros(w.shape[0])
            for k, row in enumerate(w):
                result[k] = torch.prod(row[row != 0] * d[torch.argwhere(row != 0).T])

            return result

        if self.simple_delays:
            self.output = _product_aggregation(self.weights, self.delayed_data)

        else:
            self.delayed_data = self.delayed_data.reshape(self.matrix_shape)

            # this loop is unavoidable, as N vector multiplications with different delayed_data are needed.
            for i in range(self.matrix_shape[0]):
                self.output[i] = _product_aggregation(self.weights[i, :], self.delayed_data[i, :])

        # advance simulation step and return output dictionary.
        self.simulation_step += 1

        return self.loggable_state()


class IICSynapse(Synapse):
    """Inhibitory plastic synapse for use in the Imam & Cleland (2020) model replication.

    These synapses maintain an integer matrix corresponding to the duration of inhibition that
    presynaptic units should exert postsynaptic units, in simulation steps.

    Note
    ----
    For future proofing, I avoided conflating the blocking period with the weight matrix.
    Weights are reserved for specifying the magnitude and sign of the constant synaptic output,
    anticipating modifications depending on, e.g., GC order and simplifying the rebound mechanism.

    """

    _config_props_: tuple[str] = (
        "weight_max",
        "weight_min",
        "delay_ms",
        "simple_delays",
        "learning_rate",
        "units_per_column",
        "cycle_length",
    )
    _loggable_props_: tuple[str] = ("weights", "blocking_durations", "connections", "output")

    def __init__(self, learning_rate: float = 1.0, units_per_column: int = 0, cycle_length: int = 50, **kwargs):
        super().__init__(**kwargs)

        # state variables specific to inhibitory plastic synapses, over and above Synapse.
        self.src_last_spiked = torch.zeros(self.matrix_shape[1], dtype=torch.float, device=self.device)
        self.dst_last_spiked = torch.zeros(self.matrix_shape[0], dtype=torch.float, device=self.device)

        # initialize blocking durations matrix.
        self.blocking_durations = torch.zeros_like(self.weights)

        # enable or disable learning (weight update method call).
        self.learning = True

        # the learning rate parameter eta from Imam & Cleland (2020).
        self.learning_rate = learning_rate

        # optionally, specify columnar organization of units.
        # if nonzero, nonlocal column-wise operations will be applied to consecutive segments of this size.
        self.units_per_column = units_per_column

        # length of reset cycle (e.g., gamma) in simulation steps. Must be known to bound delta B appropriately.
        self.cycle_length = cycle_length

        # keep track of which blocking durations were updated.
        self.stale = torch.zeros_like(self.blocking_durations).bool()

    def refresh_last_spiked(self):
        # update last spike time to current simulation step iff a spike has just arrived at the synapse.
        self.src_last_spiked = self.src_last_spiked + self.src_ensemble.spiked * (
            -self.src_last_spiked + self.simulation_step
        )

        # update last spike time only for postsynaptic units that spiked in this model iteration.
        self.dst_last_spiked = self.dst_last_spiked + self.dst_ensemble.spiked * (
            -self.dst_last_spiked + self.simulation_step
        )

    def update_blocking(self) -> Tensor:
        """Inhibitory plastic synapse blocking duration update rule, called at the end of each oscillatory cycle.

        Delta is the difference between the last spike times of the source and destination units,
        in simulation steps, multiplied by the learning rate Eta.

        Warning
        -------
        The mechanism assumes targets (MCs) and sources (GCs) only fire during non-overlapping windows within a cycle.
        This must be guaranteed by other parameter choices (e.g., refractory period or transmission delays).
        It also assumes sources (GCs) fire only once per cycle.

        Returns
        -------
        Tensor
            2D tensor containing blocking duration differences.

        """
        # extend last spiking timestamps of source and destination populations to enable pairwise condition checking.
        # for instance, check whether GC_i, MC_j both fired using boolean operations on these tensors.
        src_row = self.src_last_spiked.reshape(1, self.src_last_spiked.shape[0])
        src_extended = self.connections * src_row.repeat(self.dst_ensemble.num_units, src_row.shape[0])

        dst_col = self.dst_last_spiked.reshape(self.dst_last_spiked.shape[0], 1)
        dst_extended = self.connections * dst_col.repeat(dst_col.shape[1], self.src_ensemble.num_units)

        # check which source and destination neurons fired at all.
        src_ever_spiked = self.src_last_spiked != 0
        dst_ever_spiked = self.dst_last_spiked != 0

        # check which source and destination neurons fired last cycle.
        src_last_cycle = src_ever_spiked * ((self.simulation_step - self.src_last_spiked) < self.cycle_length)
        src_last_cycle = src_last_cycle.reshape(1, src_last_cycle.shape[0]).repeat(self.dst_ensemble.num_units, 1)

        dst_last_cycle = dst_ever_spiked * ((self.simulation_step - self.dst_last_spiked) < self.cycle_length)
        dst_last_cycle = dst_last_cycle.reshape(dst_last_cycle.shape[0], 1).repeat(1, self.src_ensemble.num_units)

        # only update blocking durations for pairs where both fired in this cycle, if their connection is enabled.
        to_update = self.connections * (dst_last_cycle * src_last_cycle)

        # compute spike timing differences between destination and source.
        # GC - MC is time to beginning of inhibition (from the transmission delayed perspective of MC).
        diff_spike = to_update * (src_extended - dst_extended)
        # diff_spike = diff_spike * (torch.abs(diff_spike) < self.cycle_length) * (diff_spike > 0)

        # pairs where GC fired but MC didn't in this cycle should be taken to max blocking duration.
        # pairs that aren't connected to begin with shouldn't be tampered with (anticipating neurogenesis).
        to_max_out = self.connections * src_last_cycle * ~dst_last_cycle

        # desired changes in blocking duration.
        # max out blocking periods for co-columnar pairs where source fired last cycle but destination didn't.
        # max out skips rinse cycles by checking dst_last_cycle (MCs cannot spike on rinse cycles, GCs might on 1st).
        max_out = to_max_out * self.cycle_length
        # new_blocking = self.learning_rate * torch.fmod(self.cycle_length - diff_spike, self.cycle_length) + max_out
        new_blocking = self.learning_rate * torch.fmod(dst_extended, self.cycle_length) + max_out

        # blocking durations that maxed out previously are not allowed to get back in the game.
        # new_blocking[self.blocking_durations == self.cycle_length-1] = self.cycle_length

        # if not rinse cycle, update the blocking durations and ensure they don't go under 0 or above the cycle length.
        # durations that have already been learned (!= 0) should not be updated.
        if torch.any(dst_last_cycle).int():
            new_blocking = torch.clamp(new_blocking, 0, self.cycle_length)
            temp = self.blocking_durations

            self.blocking_durations = (self.stale == 1) * self.blocking_durations + (self.stale == 0) * new_blocking
            self.stale = self.stale + torch.as_tensor(temp != new_blocking).bool()

        return self.blocking_durations

    def forward(self, data: Tensor) -> dict:
        """IIC synapse forward pass.

        Only updates the blocking duration. The native weights matrix inherited from the base class will serve to
        control the magnitude of the constant synaptic output emitted for that duration.

        Additionally, weights will temporarily flip sign when their blocking duration elapses, to simulate rebound.

        """
        # use a list of queues to track transmission delays and "release" input after the fact.
        self.delayed_data = self.queue_input(data)

        if not self.simple_delays:
            self.delayed_data = self.delayed_data.reshape(self.matrix_shape)

        # enforce weight limits.
        self.weights = torch.clamp(self.weights, self.weight_min, self.weight_max)

        # update last spiked times for source and destination ensembles.
        self.refresh_last_spiked()

        # learning can be done sensibly at the end of a cycle or on every step, application depending.
        if self.learning and self.simulation_step % self.cycle_length == self.cycle_length - 1:
            # learning in these synapses amounts to modifying the blocking duration at the end of each cycle.
            self.update_blocking()

        # given blocking durations, find out who needs to output right now and what sign the output should be.
        spike_elapsed = self.simulation_step - self.src_last_spiked
        spike_phase = torch.fmod(self.src_last_spiked, self.cycle_length)

        inclusion = (self.blocking_durations > 0) * (spike_elapsed < self.cycle_length)
        block_status = self.blocking_durations + (self.cycle_length - spike_phase) - spike_elapsed

        # if the block status for a synapse is positive (within block), it should send the weight as is (active).
        # if the block status is zero (just released), it should send a flipped weight (rebound).
        # if the block status is negative (outside window), it should send zero (inactive).
        # user should initialize these weights negative, consistent with the Imam-Cleland synapse being inhibitory.
        if self.simulation_step > 0:
            # computes a mask matrix indicating which GC->MC synapses should be actively blocking (or rebounding).
            currently_active = (block_status > 0).int() - (block_status == 0).int()
        else:
            currently_active = torch.zeros_like(self.weights)

        if self.learning:
            # Imam & Cleland (2020) suppressed the effects of these synapses during training.
            self.output *= 0.0

        else:
            # emit the weights masked by who's within a blocking or rebound period.
            # sum inputs going into each destination unit to obtain the 1D output vector.
            self.output = torch.sum(self.weights * self.connections * inclusion * currently_active, dim=1)

            L.append(torch.mean(self.output).item())
            M.append(torch.sum(self.src_ensemble.spiked).item())
            K.append(torch.sum(self.dst_ensemble.spiked).item())
            N.append(torch.mean(self.dst_ensemble.input).item())

        self.simulation_step += 1

        return self.loggable_state()
