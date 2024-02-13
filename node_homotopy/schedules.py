from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Schedule(Protocol):
    """A protocol for all objects representing a homotopy schedule.

    Such objects must be a callable that accepts an integer (current epoch) and returns a float (the corresponding homotopy parameter).
    They must also have  a max_epochs property corresponding to the maximum epoch the schedule is defined for.
    """

    def __call__(self, epoch: int) -> float:
        """Returns the value of the homotopy parameter at given epoch

        Args:
            epoch: An integer denoting the current epoch.

        Returns:
            homotopy_param: A float denoting the value of the homotopy parameter at said epoch.
        """
        return 0.0

    @property
    def max_epochs(self) -> int:
        """Gives the maximum epoch the schedule is defined for.

        Returns:
            max_epoch: An integer describing the maximum epoch for the schedule.
        """
        return 0


class StepwiseSchedule:
    """A class representing a schedule where the homotopy parameter is decremented in constant amounts, in a step-wise manner.

    It is included for pedagogical purposes. In actual training, the PowerLawStepwiseSchedule performs much much better.

    Attributes:
        decrement: Float denoting how much the homotopy parameter must be decremented at each step.
        epochs_per_step: Integer denoting the length of each step inbetween the decrements.
        values: A 1D Tensor containing the distinct values of the homotopy parameter this schedule outputs.
        epochs: A 1D Tensor containing the epoch values at which the homotopy parameter is decremented.
    """

    def __init__(self, decrement: float, epochs_per_step: int):
        """Initializes the class.

        Args:
            decrement: Float denoting how much the homotopy parameter must be decremented at each step.
            epochs_per_step: Integer denoting the length of each step inbetween the decrements.
        """
        self.decrement = decrement
        self.epochs_per_step = epochs_per_step
        self.values = self._make_values()
        self.epochs = torch.arange(len(self.values)) * epochs_per_step

    def _make_values(self):
        """Computes the distinct homotopy parameter values for the schedule.

        Returns:
            values: A 1D Tensor containing the distinct values of the homotopy parameter this schedule outputs.
        """
        values = torch.arange(1.0, -self.decrement, -self.decrement)
        values = torch.clamp(values, min=0.0, max=1.0)
        return values

    def __call__(self, epoch: int) -> float:
        """Returns the value of the homotopy parameter at given epoch

        Args:
            epoch: An integer denoting the current epoch.

        Returns:
            homotopy_param: A float denoting the value of the homotopy parameter at said epoch.
        """
        ind = torch.bucketize(epoch, self.epochs, right=True) - 1
        return self.values[ind]

    @property
    def max_epochs(self) -> int:
        """Gives the maximum epoch the schedule is defined for.

        Returns:
            max_epoch: An integer describing the maximum epoch for the schedule.
        """
        return int(self.epochs[-1] + self.epochs_per_step)


class PowerLawStepwiseSchedule:
    """A class representing a schedule where the homotopy parameter is decrementedin a step-wise, power-law manner.

    This means that after each step, the size of the decrement is multiplied by some constant positive factor.
    If this value is smaller than one, the size of the decrements become smaller and smaller with steps. This makes for a very effective training, as empirically demonstrated in the paper.

    Attributes:
        decrement: Float denoting how much the homotopy parameter must be decremented at each step.
        epochs_per_step: Integer denoting the length of each step inbetween the decrements.
        values: A 1D Tensor containing the distinct values of the homotopy parameter this schedule outputs.
        epochs: A 1D Tensor containing the epoch values at which the homotopy parameter is decremented.
        step_decay_rate: A non-negative float corresponding to the (decrement at i+1-th step) / (decrement at i-th step).
                A value smaller than one means that the size of the decrements decrease with subsequent steps.
    """

    def __init__(self, n_steps: int, epochs_per_step: int, step_decay_rate: float):
        """Initializes the class.

        Args:
            decrement: Float denoting how much the homotopy parameter must be decremented at each step.
            epochs_per_step: Integer denoting the length of each step inbetween the decrements.
            step_decay_rate: A non-negative float corresponding to the (decrement at i+1-th step) / (decrement at i-th step).
                A value smaller than one means that the size of the decrements decrease with subsequent steps.
        """
        self.n_steps = n_steps
        self.step_decay_rate = step_decay_rate
        self.epochs_per_step = epochs_per_step
        self.values = self._make_values()
        self.epochs = torch.arange(len(self.values)) * epochs_per_step

    def _make_values(self):
        """Computes the distinct homotopy parameter values for the schedule.

        Returns:
            values: A 1D Tensor containing the distinct values of the homotopy parameter this schedule outputs.
        """
        decrements = self.step_decay_rate ** torch.arange(self.n_steps - 1)
        decrements = decrements / torch.sum(decrements)

        values = torch.ones(len(decrements) + 1)
        for i in range(len(values) - 1):
            values[i + 1] = values[i] - decrements[i]
        values[-1] = 0.0
        return values

    def __call__(self, epoch: int) -> float:
        """Returns the value of the homotopy parameter at given epoch

        Args:
            epoch: An integer denoting the current epoch.

        Returns:
            homotopy_param: A float denoting the value of the homotopy parameter at said epoch.
        """
        ind = torch.bucketize(epoch, self.epochs, right=True) - 1
        return self.values[ind]

    @property
    def max_epochs(self) -> int:
        """Gives the maximum epoch the schedule is defined for.

        Returns:
            max_epoch: An integer describing the maximum epoch for the schedule.
        """
        return int(self.epochs[-1] + self.epochs_per_step)
