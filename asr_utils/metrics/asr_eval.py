import pandas as pd
import jiwer
from typing import Tuple, Union
from pathlib import Path

# TODO:
# - load queries from file
# - grouped eval?


class ASREval:
    def __init__(
        self,
        refs_file: Union[Path, str],
        hyps_file: Union[Path, str],
        key: str = "path",
        refs_col: str = "transcript",
        hyps_col: str = "hypothesis",
        join: str = "inner",
        sep: str = None,
    ):
        """Joins the files (CSV or TSV) containing the ASR hypotheses and reference transcriptions on the specified key.
        Allows WER and CER to be calculated for a subset of the data using a dataframe query.

        Args:
            refs_file (Union[Path, str]): Path to the CSV/TSV containing the references and metadata.
            hyps_file (Union[Path, str]): Path to the CSV/TSV containig the hypotheses. Should only have two columns: `key` and `hyps_col`.
            key (str, optional): Key used for joining the files' columns. Defaults to "path".
            refs_col (str, optional): Column in `refs_file` containing the references. Defaults to "transcript".
            hyps_col (str, optional): Column in `hyps_file` containing the hypotheses. Defaults to "hypothesis".
            join (str, optional): Type of join used. Defaults to "inner".
            sep (str, optional): Separator used for reading files. If none, infers the separator from file suffix (',' for CSV). Defaults to None.

        Raises:
            ValueError: Column provided does not exist in the respective TSV.
        """
        refs_file = Path(refs_file)
        hyps_file = Path(hyps_file)

        supported = set([".csv", ".tsv"])

        if not refs_file.suffix in supported or not hyps_file.suffix in supported:
            raise ValueError(
                f"File format not supported. Supported formats: {', '.join(list(supported))}"
            )

        if sep:
            refs_sep = hyps_sep = sep
        else:
            refs_sep = "," if refs_file.suffix == ".csv" else "\t"
            hyps_sep = "," if hyps_file.suffix == ".csv" else "\t"

        refs_df = pd.read_csv(refs_file, sep=refs_sep)
        hyps_df = pd.read_csv(hyps_file, sep=hyps_sep)

        if refs_col not in refs_df.columns:
            raise ValueError(f"Column `{refs_col}` not found in `{refs_file}`.")
        if hyps_col not in hyps_df.columns:
            raise ValueError(f"Column `{hyps_col}` not found in `{hyps_file}`.")

        # BHG - adding dropna to handle missing columns
        self.df = pd.merge(refs_df, hyps_df, how=join, on=key).dropna(subset=[refs_col, hyps_col)
        self.refs_col = refs_col
        self.hyps_col = hyps_col

    def eval(self) -> Tuple[float, float]:
        """Calculate error metrics for the entire dataset.
        Metrics include word-level and character-level error rate (WER, CER), substitutions, deletions, insertions, hits and counts.

        Returns:
            dict: Dict containing computed metrics
        """
        return self._compute_metrics(self.df)

    def eval_with_query(self, query: str) -> Tuple[float, float]:
        """Calculate error metrics for the subset of the data that matches the provided query.
        Metrics include word-level and character-level error rate (WER, CER), substitutions, deletions, insertions, hits and counts.

        Args:
            query (str): Query string to evaluate. Please refer to the Pandas documentation for more information: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html"

        Returns:
            dict: Dict containing computed metrics
        """
        q = self.df.query(query)
        return self._compute_metrics(q)

    def _compute_metrics(self, df: pd.DataFrame) -> Tuple[float, float]:
        refs = list(df[self.refs_col])
        hyps = list(df[self.hyps_col])

        wer_score = jiwer.wer(refs, hyps)
        cer_score = jiwer.cer(refs, hyps)

        w_out = jiwer.process_words(refs, hyps)
        c_out = jiwer.process_characters(refs, hyps)

        return {
            "wer": wer_score,
            "cer": cer_score,
            "w_sub": w_out.substitutions,
            "w_del": w_out.deletions,
            "w_ins": w_out.insertions,
            "w_hits": w_out.hits,
            "w_count": w_out.hits + w_out.substitutions + w_out.deletions,
            "c_sub": c_out.substitutions,
            "c_del": c_out.deletions,
            "c_ins": c_out.insertions,
            "c_hits": c_out.hits,
            "c_count": c_out.hits + c_out.substitutions + c_out.deletions,
        }
