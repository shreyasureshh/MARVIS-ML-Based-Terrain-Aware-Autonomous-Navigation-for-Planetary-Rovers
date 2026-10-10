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
