__all__ = ["AgainInferenceAdapter", "load_legacy_benchmark_rows"]


def __getattr__(name: str):
    if name == "AgainInferenceAdapter":
        from again_benchmark.adapters.again_inference import AgainInferenceAdapter

        return AgainInferenceAdapter
    if name == "load_legacy_benchmark_rows":
        from again_benchmark.adapters.legacy_benchmark_bridge import load_legacy_benchmark_rows

        return load_legacy_benchmark_rows
    raise AttributeError(name)
