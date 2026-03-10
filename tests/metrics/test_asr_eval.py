import pytest
from asr_utils.metrics.asr_eval import ASREval

import csv
import jiwer


@pytest.fixture
def refs_tsv(tmp_path):
    with open(tmp_path / "refs.tsv", "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(
            [["path", "transcript", "etiology"],
             ["a.wav", "hello world", "PD"],
             ["b.wav", "recognize speech", "PD"],
             ["c.wav", "what would you do?", "ALS"]]
        )
    return tmp_path / "refs.tsv"


@pytest.fixture
def refs_csv(tmp_path):
    with open(tmp_path / "refs.csv", "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(
            [["path", "transcript", "etiology"],
             ["a.wav", "hello world", "PD"],
             ["b.wav", "recognize speech", "PD"],
             ["c.wav", "what would you do?", "ALS"]]
        )
    return tmp_path / "refs.csv"


@pytest.fixture
def hyps_tsv(tmp_path):
    with open(tmp_path / "hyps.tsv", "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(
            [["path", "hypothesis"],
             ["a.wav", "hello world"],
             ["b.wav", "wreck a nice beach"],
             ["c.wav", "what would you do?"]]
        )
    return tmp_path / "hyps.tsv"


@pytest.fixture
def hyps_csv(tmp_path):
    with open(tmp_path / "hyps.csv", "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(
            [["path", "hypothesis"],
             ["a.wav", "hello world"],
             ["b.wav", "wreck a nice beach"],
             ["c.wav", "what would you do?"]]
        )
    return tmp_path / "hyps.csv"


def test_asr_eval(refs_csv, hyps_tsv):
    asr_eval = ASREval(refs_csv, hyps_tsv)

    refs = ["hello world", "recognize speech", "what would you do?"]
    hyps = ["hello world", "wreck a nice beach", "what would you do?"]

    w_out = jiwer.process_words(refs, hyps)
    c_out = jiwer.process_characters(refs, hyps)

    test_metrics = {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
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

    assert asr_eval.eval() == test_metrics


def test_asr_eval_with_query(refs_csv, hyps_csv):
    asr_eval = ASREval(refs_csv, hyps_csv)

    refs = ["hello world", "recognize speech"]
    hyps = ["hello world", "wreck a nice beach"]

    w_out = jiwer.process_words(refs, hyps)
    c_out = jiwer.process_characters(refs, hyps)

    test_metrics = {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
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

    assert asr_eval.eval_with_query("etiology == 'PD'") == test_metrics
