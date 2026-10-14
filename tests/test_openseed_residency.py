"""Verify residency changes preserve the original prediction and failure path."""
from types import SimpleNamespace

import pytest

from orbvi_occ import openseed_provider, openseed_residency


def worker_fixture(monkeypatch, *, fail=False):
    events = []
    text_cache = object()
    output = [{'original_prediction': object()}]

    class Model:
        def to(self, device):
            events.append(('to', device))
            self.device = device
            return self

    def original_init(worker, *args, **kwargs):
        events.append(('strict_original_init', args, kwargs))
        worker.model = Model()
        worker.device = 'cuda'
        worker.last_info = {}
        worker._metadata = {'weights_loaded_strict': True}
        worker.text_cache = text_cache
        worker.torch = SimpleNamespace(cuda=SimpleNamespace(
            empty_cache=lambda: events.append(('empty_cache',)),
            memory_allocated=lambda: 12 * 2**20,
            memory_reserved=lambda: 14 * 2**20))

    def original_predict(worker, views):
        assert worker.model.device == 'cuda'
        assert worker.text_cache is text_cache
        events.append(('original_predict', views))
        worker.last_info = {'original_diagnostic': True}
        if fail:
            raise RuntimeError('original inference failure')
        return output

    monkeypatch.setattr(openseed_provider.OpenSeeDWorker, '__init__', original_init)
    monkeypatch.setattr(openseed_provider.OpenSeeDWorker, 'predict_views', original_predict)
    worker = openseed_residency.ResidencyWorker('checkpoint', topk=3)
    return worker, events, output


def test_constructor_preserves_strict_loader_then_parks_registered_model(monkeypatch):
    worker, events, _ = worker_fixture(monkeypatch)
    assert events == [('strict_original_init', ('checkpoint',), {'topk': 3}),
                      ('to', 'cpu'), ('empty_cache',)]
    assert worker.device == 'cuda' and worker.model.device == 'cpu'
    assert worker.metadata()['weights_loaded_strict']


def test_repeated_predictions_return_original_objects_and_park_each_time(monkeypatch):
    worker, events, output = worker_fixture(monkeypatch)
    events.clear()
    views = [object()]
    for _ in range(2):
        assert worker.predict_views(views) is output
        assert worker.model.device == 'cpu'
    assert events == [('to', 'cuda'), ('original_predict', views), ('to', 'cpu'), ('empty_cache',)] * 2
    assert worker.last_info['original_diagnostic'] is True
    assert worker.last_info['worker_idle_allocated_gpu_mib'] == 12


def test_original_inference_failure_is_preserved_and_gpu_model_is_parked(monkeypatch):
    worker, events, _ = worker_fixture(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match='original inference failure'):
        worker.predict_views([])
    assert worker.model.device == 'cpu'
    assert events[-2:] == [('to', 'cpu'), ('empty_cache',)]


def test_entrypoint_reuses_original_protocol_without_importing_cuda(monkeypatch):
    called = []
    monkeypatch.setattr(openseed_provider, 'OpenSeeDWorker', openseed_provider.OpenSeeDWorker)
    monkeypatch.setattr(openseed_provider, 'main', lambda: called.append(openseed_provider.OpenSeeDWorker))
    openseed_residency.main()
    assert called == [openseed_residency.ResidencyWorker]
