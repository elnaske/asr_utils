import pandas as pd
from jiwer import wer, cer
from typing import Tuple, Union
from pathlib import Path


class ASREval:
    def __init__(
        self,
        refs_tsv: Union[Path, str],
        hyps_tsv: Union[Path, str],
        key: str = "path",
        refs_col: str = "transcript",
        hyps_col: str = "hypotheses",
        join: str = "inner",
    ):
        """Joins the TSV files containing the ASR hypotheses and reference transcriptions on the specified key.
        Allows WER and CER to be calculated for a subset of the data using a dataframe query.

        Args:
            refs_tsv (Union[Path, str]): Path to the TSV containing the references and metadata. 
            hyps_tsv (Union[Path, str]): Path to the TSV containig the hypotheses. Should only have two columns: `key` and `hyps_col`.
            key (str, optional): Key used for joining the TSVs. Defaults to "path".
            refs_col (str, optional): Column in `refs_tsv` containing the references. Defaults to "transcript".
            hyps_col (str, optional): Column in `hyps_tsv` containing the hypotheses. Defaults to "hypotheses".
            join (str, optional): Type of join used. Defaults to "inner".

        Raises:
            ValueError: Column provided does not exist in the respective TSV. 
        """
        refs_df = pd.read_csv(refs_tsv, sep='\t')
        hyps_df = pd.read_csv(hyps_tsv, sep='\t')

        if refs_col not in refs_df.columns:
            raise ValueError(f"Column `{refs_col}` not found in `{refs_tsv}`.")
        if hyps_col not in hyps_df.columns:
            raise ValueError(f"Column `{hyps_col}` not found in `{hyps_tsv}`.")

        self.df = pd.merge(refs_df, hyps_df, how=join, on=key)
        self.refs_col = refs_col
        self.hyps_col = hyps_col

    def eval(self) -> Tuple[float, float]:
        """Calculate WER and CER for the entire dataset.

        Returns:
            Tuple[float, float]: Tuple of WER, CER.
        """
        return self._calculate_wer_cer(self.df)

    def eval_with_query(self, query: str) -> Tuple[float, float]:
        """Calculate WER and CER for the subset of the data that matches the provided query.

        Args:
            query (str): Query string to evaluate. Please refer to the Pandas documentation for more information: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html"

        Returns:
            Tuple[float, float]: Tuple of WER, CER.
        """
        q = self.df.query(query)
        return self._calculate_wer_cer(q)

    def _calculate_wer_cer(self, df: pd.DataFrame) -> Tuple[float, float]:
        wer_score = wer(list(df[self.refs_col]), list(df[self.hyps_col]))
        cer_score = cer(list(df[self.refs_col]), list(df[self.hyps_col]))
        return (wer_score, cer_score)
