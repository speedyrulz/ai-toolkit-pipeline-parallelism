"""Scoring contract tests (spec §9.1). Run: python -m unittest toolkit.identity_eval.test_scoring"""

import unittest

from toolkit.identity_eval import scoring as S

W = {'a': 0.6, 'b': 0.4}


class TestFeatureScore(unittest.TestCase):
    def test_baseline_scores_zero(self):
        score, ratio, status = S.feature_score(0.3, 0.3, 0.8)
        self.assertEqual(score, 0.0)
        self.assertEqual(status, S.VALID)

    def test_target_scores_hundred_and_beyond_is_clipped(self):
        self.assertEqual(S.feature_score(0.8, 0.3, 0.8)[0], 100.0)
        score, ratio, _ = S.feature_score(1.3, 0.3, 0.8)
        self.assertEqual(score, 100.0)
        self.assertAlmostEqual(ratio, 2.0)  # kept unclipped for the memorization warning

    def test_below_baseline_is_zero_not_negative(self):
        self.assertEqual(S.feature_score(0.1, 0.3, 0.8)[0], 0.0)

    def test_uncalibratable_when_target_not_above_baseline(self):
        score, ratio, status = S.feature_score(0.9, 0.8, 0.8)
        self.assertIsNone(score)
        self.assertEqual(status, S.UNCALIBRATED)

    def test_missing_raw_is_failed_not_zero(self):
        score, _, status = S.feature_score(None, 0.3, 0.8)
        self.assertIsNone(score)
        self.assertEqual(status, S.FAILED)

    def test_bounds_hold_for_any_input(self):
        for raw in (-5.0, 0.0, 0.31, 0.55, 0.79, 0.8, 10.0):
            score = S.feature_score(raw, 0.3, 0.8)[0]
            self.assertTrue(0.0 <= score <= 100.0)


class TestWeightedScore(unittest.TestCase):
    def test_missing_feature_never_renormalizes(self):
        self.assertIsNone(S.weighted_score({'a': 90.0, 'b': None}, W))

    def test_weights(self):
        self.assertAlmostEqual(S.weighted_score({'a': 100.0, 'b': 50.0}, W), 80.0)


class TestFuse(unittest.TestCase):
    def test_corners(self):
        self.assertEqual(S.fuse(100.0, 100.0), 100.0)
        self.assertEqual(S.fuse(100.0, 0.0), 0.0)
        self.assertEqual(S.fuse(0.0, 100.0), 0.0)
        self.assertEqual(S.fuse(0.0, 0.0), 0.0)

    def test_missing_modality_is_none(self):
        self.assertIsNone(S.fuse(None, 80.0))
        self.assertIsNone(S.fuse(80.0, None))

    def test_spec_worked_example(self):
        # spec §6.2: F=38, B=61 -> 47.1 (the formula gives 47.02; the spec
        # rounded loosely); F=68, B=82 -> 74.0; F=92, B=75 -> 83.9
        self.assertAlmostEqual(S.fuse(38.0, 61.0), 47.1, delta=0.1)
        self.assertAlmostEqual(S.fuse(68.0, 82.0), 74.0, places=1)
        self.assertAlmostEqual(S.fuse(92.0, 75.0), 83.9, places=1)

    def test_bounded(self):
        for f in (0, 13, 50, 99, 100):
            for b in (0, 7, 50, 100):
                self.assertTrue(0.0 <= S.fuse(float(f), float(b)) <= 100.0)


class TestTargetAndAggregate(unittest.TestCase):
    def test_target_quantile(self):
        self.assertAlmostEqual(S.target([0.6, 0.7, 0.8, 0.9], 0.75), 0.825)
        self.assertIsNone(S.target([0.7], 0.75))

    def test_aggregate_median_per_bin_then_mean(self):
        slots = [
            {'bin': 'front', 'score': 80.0},
            {'bin': 'front', 'score': 60.0},
            {'bin': 'front', 'score': 70.0},
            {'bin': 'side', 'score': 40.0},
            {'bin': 'side', 'score': 50.0},
        ]
        agg = S.aggregate(slots, expected_slots=5)
        self.assertAlmostEqual(agg['score'], (70.0 + 45.0) / 2)
        self.assertEqual(agg['status'], S.VALID)

    def test_aggregate_low_coverage_is_missing(self):
        slots = [{'bin': 'front', 'score': 80.0}, {'bin': 'side', 'score': None},
                 {'bin': 'side', 'score': None}]
        agg = S.aggregate(slots, expected_slots=3)
        self.assertIsNone(agg['score'])
        self.assertEqual(agg['status'], S.MISSING_MODALITY)

    def test_step0_round_scores_zero(self):
        # paired baseline: the step-0 round scored against itself is exactly 0
        raws = {'front_0': 0.31, 'side_0': 0.22, 'back_0': 0.4}
        bins = {'front_0': 'front', 'side_0': 'side', 'back_0': 'back'}
        slots = []
        for slot, raw in raws.items():
            r = S.score_slot({'a': raw, 'b': raw}, {'a': raw, 'b': raw}, {'a': 0.8, 'b': 0.8}, W)
            slots.append({'bin': bins[slot], 'score': r['score']})
        agg = S.aggregate(slots, expected_slots=3)
        self.assertEqual(agg['score'], 0.0)


if __name__ == '__main__':
    unittest.main()
