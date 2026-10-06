"""
KALU Router Production

Charge le bundle :
    kalu_router_B1_LANG_v1.joblib

Expose :
    analyze_kalu_message(message, previous_language=None)

Dépendances :
    pip install scikit-learn scipy numpy joblib

IMPORTANT :
- Charger uniquement un fichier .joblib provenant d'une source de confiance.
- Utiliser idéalement la même version de scikit-learn qu'à l'entraînement.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

import joblib
import numpy as np
import sklearn
from scipy.sparse import hstack


# =============================================================================
# 1. CHARGEMENT DU BUNDLE
# =============================================================================

MODEL_PATH = os.getenv(
    "KALU_ROUTER_MODEL_PATH",
    "kalu_router_B1_LANG_v1.joblib",
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Bundle KALU introuvable : {MODEL_PATH}\n"
        "Définissez éventuellement la variable d'environnement "
        "KALU_ROUTER_MODEL_PATH."
    )

KALU_ROUTER = joblib.load(MODEL_PATH)


# =============================================================================
# 2. VALIDATION MINIMALE
# =============================================================================

required_sections = {
    "intent",
    "language",
    "language_policy",
    "routes",
}

missing_sections = required_sections - set(KALU_ROUTER.keys())

if missing_sections:
    raise RuntimeError(
        "Bundle KALU invalide. Sections manquantes : "
        + ", ".join(sorted(missing_sections))
    )


# =============================================================================
# 3. COMPOSANTS
# =============================================================================

INTENT_WORD_VECTORIZER = KALU_ROUTER["intent"]["word_vectorizer"]
INTENT_CHAR_VECTORIZER = KALU_ROUTER["intent"]["char_vectorizer"]
INTENT_CLASSIFIER = KALU_ROUTER["intent"]["classifier"]

LANGUAGE_VECTORIZER = KALU_ROUTER["language"]["vectorizer"]
LANGUAGE_CLASSIFIER = KALU_ROUTER["language"]["classifier"]

LANGUAGE_POLICY = KALU_ROUTER["language_policy"]
INTENT_ROUTES = KALU_ROUTER["routes"]


# =============================================================================
# 4. CONTRÔLE VERSION SCIKIT-LEARN
# =============================================================================

trained_sklearn = (
    KALU_ROUTER
    .get("environment", {})
    .get("sklearn")
)

if trained_sklearn and trained_sklearn != sklearn.__version__:
    warnings.warn(
        "Version scikit-learn différente : "
        f"entraînement={trained_sklearn}, runtime={sklearn.__version__}. "
        "Pour une production stable, utilisez idéalement la même version.",
        RuntimeWarning,
    )


# =============================================================================
# 5. UTILITAIRE TOP-2
# =============================================================================

def _top2(
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, Any]:
    order = np.argsort(probabilities)[::-1]

    if len(order) < 2:
        raise RuntimeError(
            "Le classifieur doit contenir au moins deux classes."
        )

    best_idx = int(order[0])
    second_idx = int(order[1])

    best_confidence = float(probabilities[best_idx])
    second_confidence = float(probabilities[second_idx])

    return {
        "label": str(classes[best_idx]),
        "confidence": best_confidence,
        "second_label": str(classes[second_idx]),
        "second_confidence": second_confidence,
        "margin": best_confidence - second_confidence,
    }


# =============================================================================
# 6. ANALYSE MESSAGE
# =============================================================================

def analyze_kalu_message(
    message: str,
    previous_language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyse un message entrant pour KALU.

    Parameters
    ----------
    message:
        Message utilisateur.

    previous_language:
        Langue du tour précédent : "fr" ou "en".
        Sert de fallback si le détecteur de langue est incertain.

    Returns
    -------
    dict
        intent, confidence, language, route, etc.
    """

    message = str(message).strip()

    if not message:
        raise ValueError("Le message ne peut pas être vide.")

    # -------------------------------------------------------------------------
    # A. INTENTION B1
    # -------------------------------------------------------------------------

    X_word = INTENT_WORD_VECTORIZER.transform([message])
    X_char = INTENT_CHAR_VECTORIZER.transform([message])

    X_intent = hstack(
        [X_word, X_char],
        format="csr",
    )

    intent_probabilities = INTENT_CLASSIFIER.predict_proba(
        X_intent
    )[0]

    intent_result = _top2(
        intent_probabilities,
        INTENT_CLASSIFIER.classes_,
    )

    intent = intent_result["label"]

    # -------------------------------------------------------------------------
    # B. LANGUE
    # -------------------------------------------------------------------------

    X_language = LANGUAGE_VECTORIZER.transform([message])

    language_probabilities = LANGUAGE_CLASSIFIER.predict_proba(
        X_language
    )[0]

    language_result = _top2(
        language_probabilities,
        LANGUAGE_CLASSIFIER.classes_,
    )

    raw_language = language_result["label"]
    language_confidence = language_result["confidence"]
    language_margin = language_result["margin"]

    # -------------------------------------------------------------------------
    # C. RÉSOLUTION LANGUE INCERTAINE
    # -------------------------------------------------------------------------

    min_confidence = float(
        LANGUAGE_POLICY.get("min_confidence", 0.65)
    )

    min_margin = float(
        LANGUAGE_POLICY.get("min_margin", 0.15)
    )

    fallback_language = str(
        LANGUAGE_POLICY.get("fallback_language", "fr")
    )

    language_uncertain = (
        language_confidence < min_confidence
        or language_margin < min_margin
    )

    if language_uncertain:
        if previous_language in {"fr", "en"}:
            language = previous_language
        else:
            language = fallback_language
    else:
        language = raw_language

    # -------------------------------------------------------------------------
    # D. ROUTE KALU
    # -------------------------------------------------------------------------

    route = INTENT_ROUTES.get(
        intent,
        "UNKNOWN",
    )

    prompt_language = {
        "fr": "French",
        "en": "English",
    }.get(
        language,
        "French",
    )

    return {
        "message": message,

        # Intention
        "intent": intent,
        "intent_confidence": intent_result["confidence"],
        "intent_second": intent_result["second_label"],
        "intent_second_confidence": intent_result["second_confidence"],
        "intent_margin": intent_result["margin"],

        # Langue
        "language": language,
        "prompt_language": prompt_language,
        "raw_language": raw_language,
        "language_confidence": language_confidence,
        "language_second": language_result["second_label"],
        "language_second_confidence": language_result["second_confidence"],
        "language_margin": language_margin,
        "language_uncertain": language_uncertain,

        # Routage
        "route": route,
    }


# =============================================================================
# 7. INTÉGRATION KALU
# =============================================================================
#
# decision = analyze_kalu_message(
#     user_message,
#     previous_language=session_language,
# )
#
# session_language = decision["language"]
#
# if decision["intent"] == "PRODUCT_INFORMATION":
#     response = product_rag(
#         message=user_message,
#         response_language=decision["language"],
#     )
#
# IMPORTANT :
# Le RAG ne doit PLUS redétecter lui-même la langue.
# Il doit utiliser decision["language"].
# =============================================================================


# =============================================================================
# 8. SMOKE TEST OPTIONNEL
# =============================================================================

if __name__ == "__main__":
    samples = [
        "Qu’est-ce que le FCP Coris Performance ?",
        "Who manages Coris Actions?",
    ]

    for sample in samples:
        result = analyze_kalu_message(sample)

        print("\n" + "=" * 80)
        print("MESSAGE :", sample)
        print("INTENT  :", result["intent"])
        print("CONF.   :", round(result["intent_confidence"], 4))
        print("LANGUE  :", result["language"])
        print("L-CONF. :", round(result["language_confidence"], 4))
        print("ROUTE   :", result["route"])
