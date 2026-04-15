"""
Split utilities for PyHealth SampleDataset objects.

Provides ``split_by_function``, which partitions a SampleDataset by the value
of a per-sample field (e.g. the ``"split"`` field produced by RareDis / RDD
tasks that carry an explicit train/dev/test label).

Usage::

    from splitter.splitter import split_by_function

    train, dev, test = split_by_function(
        sample_dataset,
        split_key="split",
        splits=("train", "dev", "test"),
    )
"""

from typing import Sequence, Tuple

from pyhealth.datasets.sample_dataset import SampleDataset


def split_by_function(
    dataset: SampleDataset,
    split_key: str = "split",
    splits: Sequence[str] = ("train", "dev", "test"),
) -> Tuple[SampleDataset, ...]:
    """Partition a SampleDataset by the value of a per-sample field.

    Args:
        dataset:   A ``SampleDataset`` whose samples each contain ``split_key``.
        split_key: Name of the sample field that holds the partition label
                   (default: ``"split"``).
        splits:    Ordered sequence of partition labels to extract.  A
                   ``SampleDataset`` is returned for each label, in the same
                   order.  Samples whose ``split_key`` value is not in
                   ``splits`` are silently dropped.

    Returns:
        A tuple of ``SampleDataset`` objects, one per entry in ``splits``,
        in the same order.

    Raises:
        KeyError: If any returned partition is empty and the caller wants to
                  detect that (callers should check ``len()`` themselves).

    Example::

        train, dev, test = split_by_function(
            sample_dataset,
            split_key="split",
            splits=("train", "dev", "test"),
        )
    """
    # Build index lists per split label.
    indices: dict = {label: [] for label in splits}
    for i in range(len(dataset)):
        label = dataset[i].get(split_key)
        if label in indices:
            indices[label].append(i)

    return tuple(dataset.subset(indices[label]) for label in splits)
