import polars as pl

from src.data import synthetic_audit_pipeline as module
import src.data.lineage_decorator as lineage_decorator


class FakeOpenLineageClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.events = []
        type(self).instances.append(self)

    def emit(self, event):
        self.events.append(event)


class FakeCleanDataGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self):
        return pl.DataFrame({"id": [1, 2], "amount": [10.0, 20.0]})


class FakeFinancialAnomalyInjector:
    def __init__(self, config):
        self.config = config

    def inject(self, df):
        return df.with_columns(pl.Series("anomaly_flag", [False, True]))

    def get_anomaly_summary(self, df):
        return {"total_rows": len(df), "anomaly_ratio": 0.5}


def test_run_pipeline_emits_lineage_events(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "CleanDataGenerator", FakeCleanDataGenerator)
    monkeypatch.setattr(module, "FinancialAnomalyInjector", FakeFinancialAnomalyInjector)
    monkeypatch.setattr("src.data.lineage_decorator.OpenLineageClient", FakeOpenLineageClient)

    result = module.run_pipeline(
        config=module.GeneratorConfig(num_records=2, num_entities=2, seed=42),
        output_dir=tmp_path,
    )

    assert result["clean_output_path"].exists()
    assert result["anomalous_output_path"].exists()
    assert result["summary_path"].exists()

    assert FakeOpenLineageClient.instances
    assert any(instance.events for instance in FakeOpenLineageClient.instances)


def test_lineage_decorator_uses_configured_http_transport(monkeypatch):
    class ConfigurableOpenLineageClient:
        instances = []

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.events = []
            type(self).instances.append(self)

        def emit(self, event):
            self.events.append(event)

    monkeypatch.setattr(lineage_decorator, "OpenLineageClient", ConfigurableOpenLineageClient)
    monkeypatch.setenv("OPENLINEAGE_URL", "http://marquez:5000")
    monkeypatch.setattr(lineage_decorator.settings, "OPENLINEAGE_URL", "http://marquez:5000", raising=False)

    @lineage_decorator.lineage_trace(
        job_name="demo_job",
        input_datasets=["input"],
        output_datasets=["output"],
        namespace="demo.namespace",
    )
    def demo_job():
        return "ok"

    assert demo_job() == "ok"
    assert ConfigurableOpenLineageClient.instances
    assert ConfigurableOpenLineageClient.instances[0].kwargs.get("transport") is not None
