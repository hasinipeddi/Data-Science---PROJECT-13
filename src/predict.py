"""
predict.py
==========
Single-image prediction utilities for the Streamlit demo and notebook.

Functions
---------
predict_single(model, image_vec, model_type)  - Returns (digit, confidence, scores)
get_confidence_label(model_type)              - Returns appropriate label string
"""

import numpy as np


def predict_single(model, image_vec: np.ndarray, model_type: str = "knn") -> tuple:
    """Run inference on a single preprocessed 28×28 image.

    Parameters
    ----------
    model      : fitted sklearn estimator (KNeighborsClassifier or SVC)
    image_vec  : ndarray  shape (1, 784) normalised + flattened
    model_type : str  "knn" or "svm"

    Returns
    -------
    predicted_digit : int
    confidence      : float  in [0, 1] — see notes below
    scores          : ndarray  shape (10,)  raw score for each class

    Notes on confidence
    -------------------
    KNN  : fraction of the k nearest neighbours that voted for the winning class.
           E.g. k=3, 2 neighbours voted '5' → confidence = 2/3 ≈ 0.667.
    SVM  : decision_function() returns a raw score per class via one-vs-rest
           decomposition.  We apply softmax to convert scores to a
           pseudo-probability distribution in [0,1] summing to 1.
    """
    model_type = model_type.lower()

    if model_type == "knn":
        # Neighbours for this sample
        neigh_dist, neigh_idx = model.kneighbors(image_vec)
        neigh_labels = model._y[neigh_idx[0]]
        classes = model.classes_

        # Official sklearn prediction (preserves accurate distance tie-breaking)
        predicted_digit = int(model.predict(image_vec)[0])

        # Vote counts for each class
        vote_counts = np.zeros(len(classes), dtype=int)
        for label in neigh_labels:
            vote_counts[np.where(classes == label)[0][0]] += 1

        predicted_class_idx = np.where(classes == predicted_digit)[0][0]
        confidence = float(vote_counts[predicted_class_idx]) / model.n_neighbors

        # Normalised vote share as scores
        scores = vote_counts / model.n_neighbors

    elif model_type == "svm":
        # decision_function → shape (1, 10) for multi-class OvR
        raw_scores = model.decision_function(image_vec)[0]  # shape (10,)
        predicted_digit = int(model.predict(image_vec)[0])

        # Softmax for pseudo-probabilities
        exp_s = np.exp(raw_scores - raw_scores.max())
        scores = exp_s / exp_s.sum()
        confidence = float(scores[predicted_digit])

    else:
        raise ValueError(f"model_type must be 'knn' or 'svm', got '{model_type}'")

    return predicted_digit, confidence, scores


def get_confidence_label(model_type: str) -> str:
    """Return a human-readable label for the confidence metric."""
    model_type = model_type.lower()
    if model_type == "knn":
        return "Neighbour Vote Share"
    elif model_type == "svm":
        return "Decision Margin (Softmax Normalized)"
    return "Confidence"


def predict_with_details(model, image_vec: np.ndarray, model_type: str = "knn") -> dict:
    """Run inference and return detailed model diagnostics.

    Parameters
    ----------
    model      : fitted estimator (KNeighborsClassifier or SVC)
    image_vec  : ndarray shape (1, 784)
    model_type : str 'knn' or 'svm'

    Returns
    -------
    dict with diagnostic fields for dashboard visualization.
    """
    model_type = model_type.lower()
    digit, conf, scores = predict_single(model, image_vec, model_type)

    details = {
        "model_type": model_type,
        "digit": digit,
        "confidence": conf,
        "scores": scores,
    }

    if model_type == "knn":
        neigh_dist, neigh_idx = model.kneighbors(image_vec)
        neigh_labels = model._y[neigh_idx[0]].tolist()
        neigh_dists = [round(float(d), 3) for d in neigh_dist[0]]
        details["neigh_labels"] = neigh_labels
        details["neigh_dists"] = neigh_dists
        details["vote_summary"] = f"{int(round(conf * model.n_neighbors))}/{model.n_neighbors} votes ({conf*100:.0f}%)"
    elif model_type == "svm":
        raw_scores = model.decision_function(image_vec)[0]
        details["raw_scores"] = raw_scores
        details["winning_margin"] = float(raw_scores[digit])
        details["margin_summary"] = f"Margin: {raw_scores[digit]:+.2f} (Softmax: {conf*100:.1f}%)"

    return details


def compare_models(knn_model, svm_model, image_vec: np.ndarray) -> dict:
    """Run both KNN and SVM on the exact same preprocessed input and compare.

    Parameters
    ----------
    knn_model : KNeighborsClassifier
    svm_model : SVC
    image_vec : ndarray shape (1, 784)

    Returns
    -------
    dict with keys:
        knn         : detailed result for KNN
        svm         : detailed result for SVM
        agreement   : bool (True if both predict same digit)
        consensus_digit : int or None
    """
    knn_info = predict_with_details(knn_model, image_vec, "knn")
    svm_info = predict_with_details(svm_model, image_vec, "svm")

    agreement = (knn_info["digit"] == svm_info["digit"])
    consensus_digit = knn_info["digit"] if agreement else None

    return {
        "knn": knn_info,
        "svm": svm_info,
        "agreement": agreement,
        "consensus_digit": consensus_digit,
    }
