"""Orchestration-only OpenSeeD worker with CPU residency between predictions.

The original worker still loads the checkpoint strictly, constructs its seeded
text embeddings once, and runs its complete CUDA inference unchanged. Only
registered model tensors move between devices; plain CUDA text/evidence caches
retain their original contents and device semantics.
"""
from __future__ import annotations

from . import openseed_provider as _provider


class ResidencyWorker(_provider.OpenSeeDWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._park_model()
        self._metadata['gpu_residency_policy'] = (
            'Registered model parameters and buffers reside on CPU between '
            'predictions; original CUDA inference, dtypes and text caches retained')

    def _park_model(self):
        self.model.to('cpu')
        self.torch.cuda.empty_cache()

    def predict_views(self, views):
        try:
            self.model.to(self.device)
            return super().predict_views(views)
        finally:
            self._park_model()
            self.last_info['worker_idle_allocated_gpu_mib'] = (
                self.torch.cuda.memory_allocated() / 2**20)
            self.last_info['worker_idle_reserved_gpu_mib'] = (
                self.torch.cuda.memory_reserved() / 2**20)


def main():
    # Reuse the exact original stdin/stdout serving protocol and argument parser.
    # This assignment applies only inside this dedicated worker process.
    _provider.OpenSeeDWorker = ResidencyWorker
    _provider.main()


if __name__ == '__main__':
    main()
