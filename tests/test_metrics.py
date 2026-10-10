import numpy as np

from marvis.metrics import auroc, confusion_matrix, expected_calibration_error, risk_coverage, segmentation_scores


def test_confusion_and_iou_ignore():
    gt = np.array([0, 0, 1, 1, 255])
    pr = np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix(gt, pr, 2, 255)
    assert cm.tolist() == [[1, 1], [0, 2]]
    s = segmentation_scores(cm, ["a", "b"])
    assert np.isclose(s["per_class"]["a"]["IoU"], 0.5)
    assert np.isclose(s["per_class"]["b"]["IoU"], 2 / 3)
    assert np.isclose(s["pixel_accuracy"], 0.75)


def test_auroc_perfect_and_random():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5


def test_ece_perfectly_calibrated_is_zero():
    conf = np.full(1000, 0.8)
    correct = np.r_[np.ones(800), np.zeros(200)]
    assert expected_calibration_error(conf, correct)["ECE"] < 1e-9


def test_risk_coverage_oracle():
    err = np.array([0, 0, 1, 1])
    rc = risk_coverage(np.array([0.1, 0.2, 0.8, 0.9]), err)
    assert np.isclose(rc["AURC"], rc["AURC_oracle"])


def test_iou_absent_class_is_none_and_excluded_from_miou():
    cm = confusion_matrix(np.array([0, 0, 1]), np.array([0, 0, 1]), 3, 255)
    s = segmentation_scores(cm, ["a", "b", "c"])
    assert s["per_class"]["c"]["IoU"] is None or s["per_class"]["c"]["gt_pixels"] == 0
    assert np.isclose(s["mIoU"], 1.0)


def test_all_ignored_gives_empty_confusion():
    cm = confusion_matrix(np.full(5, 255), np.zeros(5, int), 2, 255)
    assert cm.sum() == 0
    assert np.isnan(segmentation_scores(cm, ["a", "b"])["mIoU"])


def test_auroc_undefined_with_one_outcome():
    assert np.isnan(auroc(np.array([0.1, 0.9]), np.array([1, 1])))


def test_auroc_inverted_is_zero():
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0


def test_ece_overconfident():
    conf = np.full(100, 0.9); correct = np.r_[np.ones(50), np.zeros(50)]
    assert np.isclose(expected_calibration_error(conf, correct)["ECE"], 0.4)


def test_uncertainty_evaluable_guards():
    from marvis.evaluate import uncertainty_evaluable
    assert not uncertainty_evaluable(np.zeros(100, bool))[0]            # too few pixels
    assert not uncertainty_evaluable(np.zeros(50_000, bool))[0]         # no errors
    e = np.zeros(50_000, bool); e[:5000] = True
    assert uncertainty_evaluable(e)[0]
