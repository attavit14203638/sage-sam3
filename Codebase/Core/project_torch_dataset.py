"""SAM3 dataset wrapper that supports gradient accumulation over micro-batches.

The wrapper groups consecutive micro-batches so that one loader item carries every micro-batch of
an optimiser step.
"""

from typing import Callable, Iterable, Optional

from torch.utils.data import DataLoader, default_collate

from sam3.train.data.torch_dataset import TorchDataset


class _AccumulatingCollate:
    def __init__(self, collate_fn: Optional[Callable], batch_size: int, steps: int) -> None:
        self.collate_fn = collate_fn or default_collate
        self.batch_size = batch_size
        self.steps = steps

    def __call__(self, samples):
        return [
            self.collate_fn(samples[start : start + self.batch_size])
            for start in range(0, len(samples), self.batch_size)
        ]


class ProjectTorchDataset(TorchDataset):
    """SAM3 dataset whose loader can group micro-batches for gradient accumulation."""
    def __init__(self, *args, gradient_accumulation_steps: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def get_loader(self, epoch) -> Iterable:
        """Return the loader for `epoch`; with `gradient_accumulation_steps` above 1, each item is a list of that many micro-batches."""
        if self.gradient_accumulation_steps <= 1:
            return super().get_loader(epoch)

        if self.sampler:
            self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size * self.gradient_accumulation_steps,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=self.sampler,
            collate_fn=_AccumulatingCollate(
                self.collate_fn,
                self.batch_size,
                self.gradient_accumulation_steps,
            ),
            worker_init_fn=self.worker_init_fn,
        )
