#!/usr/bin/env python3
"""Suite of reusable metrics for evaluating ASR performance."""

from typing import Dict, List

import jiwer


class ASRMetrics:
    """Provides static methods for ASR metrics."""

    @staticmethod
    def compute_wer(predictions: List[str], references: List[str]) -> Dict:
        """
        Compute Word Error Rate and related metrics.

        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions

        Returns:
            Dictionary with WER, CER, insertions, deletions, substitutions
        """
        wer = jiwer.wer(references, predictions)
        cer = jiwer.cer(references, predictions)

        # Get detailed measures using process_words
        output = jiwer.process_words(references, predictions)

        return {
            "wer": wer * 100,  # Convert to percentage
            "cer": cer * 100,
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
            "hits": output.hits,
            "num_words": output.hits + output.substitutions + output.deletions,
        }
