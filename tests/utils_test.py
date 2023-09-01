from src.utils import DeviceAwareModuleMixin
import torch as th


class DeviceAware(DeviceAwareModuleMixin, th.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = th.nn.Linear(1, 1)


def test_device_aware_module_mixin():
    assert DeviceAware().to(th.device("cpu")).device == th.device("cpu")
    if th.cuda.is_available():
        assert DeviceAware().to(th.device("cuda")).device.type == th.device("cuda").type
