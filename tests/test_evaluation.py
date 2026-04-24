"""Stub tests for src.evaluation.evaluator."""
import pytest
from src.evaluation.evaluator import compute_metrics, AGG_CSV_COLUMNS

def test_compute_metrics_f1():
    result = compute_metrics(tp=8, fp=2, fn=2)
    assert abs(result["precision"] - 0.8) < 1e-6
    assert abs(result["recall"] - 0.8) < 1e-6
    assert abs(result["f1"] - 0.8) < 1e-6

def test_agg_csv_columns_complete():
    assert "SPL_SET_ID" in AGG_CSV_COLUMNS
    assert "final_concept_id" in AGG_CSV_COLUMNS
    assert "postcoord_expression" in AGG_CSV_COLUMNS
