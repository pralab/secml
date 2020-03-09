from secml.ml.classifiers import CClassifierSVM


def _test_model_clf():
    """Model for testing `load_model` functionality.

    Pre-saved state will set "C=100" so that we can check
    if state is restored correctly.

    """
    return CClassifierSVM()
