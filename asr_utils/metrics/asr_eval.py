import pandas as pd
import jiwer
from typing import Optional, Tuple, Union
from pathlib import Path
import tempfile

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
        ref_trn: Optional[Union[Path, str]] = None,
        hyp_trn: Optional[Union[Path, str]] = None,
        tmp_dir: Optional[Union[Path, str]] = None,
        sclite: bool = False,
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
            ref_trn (Optional[Union[Path, str]]): Path to reference TRN file for SClite eval.
            hyp_trn (Optional[Union[Path, str]]): Path to hypothesis TRN file for SClite eval.
            tmp_dir (Optional[Union[Path, str]]): Sclite requires building a bunch of stuff. Where to create temp dir to build this all in.
            sclite (bool) : Use Sclite if True, else Jiwer. Defaults to False (Jiwer).

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

        # adding the utt id for trn building if sclite inference
        refs_df["utt_id"] = [
            f"utt{i:06d}" for i in range(1, len(refs_df) + 1)
        ]  # + 1 cause ids start at 1 with sclite

        # BHG - adding dropna to handle missing columns
        merged = pd.merge(refs_df, hyps_df, how=join, on=key).dropna(
            subset=[refs_col, hyps_col]
        )
        merged[refs_col] = merged[refs_col].astype(str).str.strip()
        merged[hyps_col] = merged[hyps_col].astype(str).str.strip()
        self.df = merged[(merged[refs_col] != "") & (merged[hyps_col] != "")].copy()

        self.refs_col = refs_col
        self.hyps_col = hyps_col
        self.key = key

        # buildin TRNs for sclite eval
        if sclite:
            # need all 3
            if ref_trn and hyp_trn and tmp_dir:
                self.sclite = True
                self.ref_trn = Path(ref_trn)
                self.hyp_trn = Path(hyp_trn)
                # need to build temp directory to build sclite files
                self.tmp_dir = Path(tmp_dir)
                self.tmp_dir.mkdir(parents=True, exist_ok=True)

                self.ref_map = self...

            else:
                raise ValueError(
                    {
                        f"For SClite evaluation, .trn files are needed. Instead got\nref_trn:\t{ref_trn}\nhyp_trn:\t{hyp_trn}\ntmp_dir:\t{tmp_dir}\n"
                    }
                )

        else:
            self.sclite = False

    def eval(self) -> Tuple[float, float]:
        """Calculate error metrics for the entire dataset.
        Metrics include word-level and character-level error rate (WER, CER), substitutions, deletions, insertions, hits and counts.

        Returns:
            dict: Dict containing computed metrics
        """
        if self.sclite:
            return self._compute_metrics_sclite(self.df)
        else:
            return self._compute_metrics_jiwer(self.df)

    def eval_with_query(self, query: str) -> Tuple[float, float]:
        """Calculate error metrics for the subset of the data that matches the provided query.
        Metrics include word-level and character-level error rate (WER, CER), substitutions, deletions, insertions, hits and counts.

        Args:
            query (str): Query string to evaluate. Please refer to the Pandas documentation for more information: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html"

        Returns:
            dict: Dict containing computed metrics
        """
        q = self.df.query(query)
        if self.sclite:
            return self._compute_metrics_sclite(q)
        else:
            return self._compute_metrics_jiwer(q)

    def _compute_metrics_jiwer(self, df: pd.DataFrame) -> Tuple[float, float]:
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

    def _compute_metrics_sclite(self, df: pd.DataFrame) -> Tuple[float, float]:

        utt_ids = list(df["utt_id"])

        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as tmpdir:
            tmpdir = Path(tmpdir)
            ref_split = tmpdir / "ref.trn"
            hyp_split = tmpdir / "hyp.trn"
            sgml_out = tmpdir / "out.sgml"

            ifself...
