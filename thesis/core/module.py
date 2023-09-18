from typing import Any
import torch as th


class BaseModule(th.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def device(self) -> th.device:
        """
        The device the module is on.

        :return: The device.
        """
        if isinstance(self, th.nn.Module):
            return next(self.parameters()).device
        raise NotImplementedError("Only modules that have parameters are supported.")

    def training_step(self, *args, **kwargs) -> dict[str, Any]:
        return {}