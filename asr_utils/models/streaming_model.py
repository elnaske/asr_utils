import lightning as L
import csv
from typing import Callable
from pathlib import Path


class StreamingASRModel(L.LightningModule):
    def __init__(self, model_class: Callable, logging_dir: str, chunk_size: int):
        super().__init__()
        self.model = model_class()
        required_methods = ["set_partial_callback",
                            "reset", "accept_chunk", "input_finished"]
        missing_methods = [
            f"{method}()" for method in required_methods if not hasattr(self.model, method)]

        if missing_methods:
            raise NotImplementedError(
                f"Model is missing the following required methods: {", ".join(missing_methods)}")

        self.model.set_partial_callback(lambda x: None)
        self.chunk_size = chunk_size

        self.test_hyps_tsv = Path(logging_dir) / "test_hyps.tsv"
        self.test_hyps_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.test_hyps_tsv, "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["path", "hypothesis"])
    
    def forward(self, x):
        self.model.reset()
        x = x.squeeze()  # single-dim
        for start in range(0, len(x), self.chunk_size):
            chunk = x[start: start + self.chunk_size]
            self.model.accept_chunk(chunk)
        return self.model.input_finished()

    def predict_step(self, batch, batch_idx):
        hyps = self(batch["audio"])

        with open(self.test_hyps_tsv, "a") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(batch["keys"], hyps))

        return hyps

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()
