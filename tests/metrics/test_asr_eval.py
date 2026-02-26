import pytest
from asr_utils.metrics.asr_eval import ASREval

import csv
from jiwer import wer, cer


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
def hyps_tsv(tmp_path):
    with open(tmp_path / "hyps.tsv", "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(
            [["path", "hypotheses"],
             ["a.wav", "hello world"],
             ["b.wav", "wreck a nice beach"],
             ["c.wav", "what would you do?"]]
        )
    return tmp_path / "hyps.tsv"


def test_asr_eval(hyps_tsv, refs_tsv):
    asr_eval = ASREval(refs_tsv, hyps_tsv)

    refs = ["hello world", "recognize speech", "what would you do?"]
    hyps = ["hello world", "wreck a nice beach", "what would you do?"]

    assert asr_eval.eval() == (wer(refs, hyps), cer(refs, hyps))

def test_asr_eval_with_query(hyps_tsv, refs_tsv):
    asr_eval = ASREval(refs_tsv, hyps_tsv)

    refs = ["hello world", "recognize speech"]
    hyps = ["hello world", "wreck a nice beach"]
    
    assert asr_eval.eval_with_query("etiology == 'PD'") == (wer(refs, hyps), cer(refs, hyps))
