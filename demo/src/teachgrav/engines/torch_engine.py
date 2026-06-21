from .base import BaseEngine


def _ensure_torch_array_api(torch):
    """Add minimal Array API hooks expected by this codebase."""
    if getattr(torch, "_teachgrav_array_api_patched", False):
        return

    if not hasattr(torch, "bool_"):
        torch.bool_ = torch.bool

    def _torch_array(data):
        if isinstance(data, (list, tuple)) and any(
            torch.is_tensor(x) for x in data
        ):
            tensors = [
                x if torch.is_tensor(x) else torch.as_tensor(x) for x in data
            ]
            return torch.stack(tensors)
        return torch.as_tensor(data)

    torch.array = _torch_array

    if not hasattr(torch.Tensor, "__array_namespace__"):

        def __array_namespace__(self, api_version=None):
            return torch

        torch.Tensor.__array_namespace__ = __array_namespace__

    if not hasattr(torch.Tensor, "astype"):

        def astype(self, dtype):
            return self.to(dtype=dtype)

        torch.Tensor.astype = astype

    torch._teachgrav_array_api_patched = True


class TorchBaseEngine(BaseEngine):
    """Shared base for all PyTorch engine variants."""

    def _setup(self):
        import torch

        _ensure_torch_array_api(torch)
        self.np = torch

    def seed_random(self, seed):
        import torch

        self.random = torch.Generator()
        if seed is not None:
            self.random.manual_seed(seed)

    def array(self, data):
        return self.np.array(data)

    def random_array(self, shape, min=0.0, max=1.0):
        res = (
            self.np.rand(size=shape, generator=self.random) * (max - min) + min
        )
        return self._move_to_device(res)

    def _move_to_device(self, tensor):
        return tensor


class TorchCpuEngine(TorchBaseEngine):
    """PyTorch on CPU."""

    pass


class TorchGpuEngine(TorchBaseEngine):
    """PyTorch on CUDA GPU."""

    def array(self, data):
        return super().array(data).to("cuda")

    def _move_to_device(self, tensor):
        return tensor.to("cuda")


class TorchMpsEngine(TorchBaseEngine):
    """PyTorch on Apple Metal Performance Shaders."""

    def array(self, data):
        res = super().array(data)
        if res.dtype == self.np.float64:
            res = res.to(dtype=self.np.float32)
        return res.to("mps")

    def _move_to_device(self, tensor):
        return tensor.to("mps")
