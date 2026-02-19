"""
Data loading utilities for ASR fine-tuning.

Supports:
- HuggingFace datasets (Common Voice, LibriSpeech, etc.)
- Custom CSV datasets
- Local audio files
- Local datasets saved with save_to_disk()
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from datasets import Audio, Dataset, DatasetDict, load_from_disk


class ASRDataloader:
    """
    Flexible data loader for fine-tuning.

    Supports multiple dataset formats and provides unified interface.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, sampling_rate: int = 16000
    ):
        """
        Initialize data loader.

        Args:
            config: Configuration dictionary (optional)
            sampling_rate: Target audio sampling rate (default: 16000)
        """
        self.config = config or {}
        self.sampling_rate = sampling_rate

    def load_from_csv(
        self,
        train_csv: str,
        test_csv: str,
        audio_column: str = "audio_path",
        text_column: str = "transcription",
        base_path: Optional[str] = None,
    ) -> DatasetDict:
        """
        Load dataset from CSV files.

        CSV format:
            audio_path, transcription
            /path/to/audio1.wav, "hello world"
            /path/to/audio2.wav, "this is a test"

        Args:
            train_csv: Path to training CSV file
            test_csv: Path to test CSV file
            audio_column: Name of column containing audio file paths
            text_column: Name of column containing transcriptions
            base_path: Base path for relative audio paths (optional)

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        print("\nLoading dataset from CSV:")
        print(f"  Train: {train_csv}")
        print(f"  Test: {test_csv}")

        # Read CSVs
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        # Adjust paths if base_path provided
        if base_path:
            base_path = Path(base_path)
            train_df[audio_column] = train_df[audio_column].apply(
                lambda p: str(base_path / p) if not Path(p).is_absolute() else p
            )
            test_df[audio_column] = test_df[audio_column].apply(
                lambda p: str(base_path / p) if not Path(p).is_absolute() else p
            )

        # Rename columns to standard names
        train_df = train_df.rename(
            columns={audio_column: "audio", text_column: "sentence"}
        )
        test_df = test_df.rename(
            columns={audio_column: "audio", text_column: "sentence"}
        )

        # Convert to datasets
        train_dataset = Dataset.from_pandas(train_df[["audio", "sentence"]])
        test_dataset = Dataset.from_pandas(test_df[["audio", "sentence"]])

        # Cast audio column to Audio type
        train_dataset = train_dataset.cast_column(
            "audio", Audio(sampling_rate=self.sampling_rate)
        )
        test_dataset = test_dataset.cast_column(
            "audio", Audio(sampling_rate=self.sampling_rate)
        )

        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

        print("\nDataset loaded:")
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Test samples: {len(test_dataset):,}")

        return dataset_dict

    def load_local(self, path: str, text_column: str = "transcript") -> DatasetDict:
        """
        Load dataset from local directory (saved with save_to_disk()).

        Args:
            path: Path to dataset directory
            text_column: Name of text column (default: 'transcript')

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        print(f"\nLoading local dataset from: {path}")

        dataset = load_from_disk(path)

        # Check if it's already a DatasetDict or a single Dataset
        if isinstance(dataset, DatasetDict):
            # Already has train/test splits
            print("\nDataset loaded:")
            print(f"  Train samples: {len(dataset['train']):,}")
            print(f"  Test samples: {len(dataset['test']):,}")

            # Rename text column if needed
            if (
                text_column != "sentence"
                and text_column in dataset["train"].column_names
            ):
                dataset = DatasetDict(
                    {
                        "train": dataset["train"].rename_column(
                            text_column, "sentence"
                        ),
                        "test": dataset["test"].rename_column(text_column, "sentence"),
                    }
                )

        else:
            # Single dataset - should have been split already
            raise ValueError(
                f"Expected DatasetDict with 'train' and 'test' splits, got {type(dataset)}. "
                "Please create train/test split before saving to disk."
            )

        # Cast audio to target sampling rate if needed
        if "audio" in dataset["train"].column_names:
            dataset = dataset.cast_column(
                "audio", Audio(sampling_rate=self.sampling_rate)
            )

        return dataset

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ASRDataloader":
        """
        Create data loader from configuration dictionary.

        Args:
            config: Configuration dictionary with 'dataset' key

        Returns:
            Initialized ASRDataloader
        """
        dataset_config = config.get("dataset", {})
        sampling_rate = config.get("audio", {}).get("sampling_rate", 16000)

        return cls(config=config, sampling_rate=sampling_rate)

    def load_dataset(self) -> DatasetDict:
        """
        Load dataset based on configuration.

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        dataset_config = self.config.get("dataset", {})
        dataset_type = dataset_config.get("type", "common_voice")
        dataset_name = dataset_config.get("name", "")

        if dataset_type == "csv":
            return self.load_from_csv(
                train_csv=dataset_config.get("train_csv"),
                test_csv=dataset_config.get("test_csv"),
                audio_column=dataset_config.get("audio_column", "audio_path"),
                text_column=dataset_config.get("text_column", "transcription"),
                base_path=dataset_config.get("base_path"),
            )
        elif dataset_type == "local":
            return self.load_local(
                path=dataset_config.get("path"),
                text_column=dataset_config.get("text_column", "transcript"),
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def prepare_dataset(
        self, dataset: Dataset, processor, text_column: str = "sentence"
    ) -> Dataset:
        """
        Prepare dataset for training (feature extraction and tokenization).

        Args:
            dataset: Input dataset with 'audio' and text columns
            processor: HuggingFace processor (feature extractor + tokenizer)
            text_column: Name of text column

        Returns:
            Processed dataset with input_values, labels, and duration
        """

        def prepare_example(batch):
            # Process audio
            audio = batch["audio"]

            inputs = processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt",
            )
            if hasattr(inputs, "input_values"):
                batch["input_features"] = inputs.input_values[0]
            elif hasattr(inputs, "input_features"):
                batch["input_features"] = inputs.input_features[0]
            else:
                raise ValueError("Unknown processor output type")

            # Tokenize text (no BOS, add EOS)
            labels = processor.tokenizer(
                batch[text_column], add_special_tokens=True
            ).input_ids
            # batch["labels"] = labels + [2]  # Add EOS token
            batch["labels"] = labels

            # Store audio duration for curriculum filtering and length bucketing
            batch["duration"] = len(audio["array"]) / audio["sampling_rate"]
            batch["input_length"] = len(audio["array"])

            return batch

        print("\nPreparing dataset (feature extraction + tokenization)...")

        prepared = dataset.map(
            prepare_example,
            remove_columns=dataset.column_names,
            num_proc=self.config.get("preprocessing", {}).get("num_proc", 4),
        )

        print(f"  Processed {len(prepared):,} samples")

        return prepared

    def filter_by_duration(
        self, dataset: Dataset, max_duration: float = 30.0, min_duration: float = 0.1
    ) -> Dataset:
        """
        Filter dataset by audio duration.

        Args:
            dataset: Input dataset with 'duration' column
            max_duration: Maximum duration in seconds
            min_duration: Minimum duration in seconds

        Returns:
            Filtered dataset
        """
        # Determine which duration column exists
        duration_col = (
            "duration" if "duration" in dataset.column_names else "audio_duration"
        )

        def is_valid_duration(duration):
            # When using input_columns, we receive the column value directly
            return min_duration <= duration <= max_duration

        # Use input_columns to avoid decoding audio
        filtered = dataset.filter(is_valid_duration, input_columns=[duration_col])

        print(f"\nFiltering by duration ({min_duration}s - {max_duration}s):")
        print(f"  Original: {len(dataset):,} samples")
        print(f"  Filtered: {len(filtered):,} samples")
        print(f"  Removed: {len(dataset) - len(filtered):,} samples")

        return filtered
