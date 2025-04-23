""" Custom neuron models. """
import torch
from torch import Tensor

from sapicore.engine.neuron.analog import AnalogNeuron
from sapicore.engine.neuron.spiking.LIF import LIFNeuron
from sapicore.engine.ensemble import Ensemble

__all__ = ("MultNeuron", "MultEnsemble")


class CyclicLIF(LIFNeuron):
    """LIF neuron that clamps its voltage to resting during a prescribed prohibitive window given a cycle length."""

    def __init__(self, cycle_length: int = 50, prohibitive_start: int = 25, **kwargs):
        super().__init__(**kwargs)

        self.cycle_length = cycle_length
        self.prohibitive = [prohibitive_start, (prohibitive_start + (cycle_length // 2)) - 1 % cycle_length]

        if self.prohibitive[0] > self.prohibitive[1]:
            self.prohibitive[0], self.prohibitive[1] = self.prohibitive[1], self.prohibitive[0]

    def forward(self, data: Tensor) -> dict:
        # prevent neuron from emitting a spike event if within prohibitive window.
        if self.prohibitive[0] <= self.simulation_step % self.cycle_length <= self.prohibitive[1]:
            self.voltage = self.volt_rest

        super().forward(data)

        return self.loggable_state()


class CyclicLIFEnsemble(Ensemble, CyclicLIF):
    """Ensemble of cyclic LIF neurons."""

    def __init__(self, **kwargs):
        # invoke parent constructor and initialization method(s).
        super().__init__(**kwargs)


class MultNeuron(AnalogNeuron):
    """Analog neuron with multiplicative aggregation of synaptic inputs from different sources."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # voltage must be initialized to identity, 1 in this case.
        self.voltage = torch.ones_like(self.voltage)

    @staticmethod
    def aggregate(inputs: list[Tensor], identifiers: list[str] = None) -> Tensor:
        return torch.prod(torch.vstack(inputs), dim=0)


class MultEnsemble(Ensemble, MultNeuron):
    """Ensemble of multiplicative analog neurons."""

    def __init__(self, **kwargs):
        # invoke parent constructor and initialization method(s).
        super().__init__(**kwargs)
