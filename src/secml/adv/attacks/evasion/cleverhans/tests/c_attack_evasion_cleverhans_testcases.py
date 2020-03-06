from secml.adv.attacks.evasion.tests import CAttackEvasionTestCases

from secml.utils import fm

IMAGES_FOLDER = fm.join(fm.abspath(__file__), 'test_images')
if not fm.folder_exist(IMAGES_FOLDER):
    fm.make_folder(IMAGES_FOLDER)


class CAttackEvasionCleverhansTestCases(CAttackEvasionTestCases):
    """Unittests interface for CAttackEvasionCleverhans."""
    images_folder = IMAGES_FOLDER

    def _test_confidence(self, x0, y0, x_opt, clf, y_target):
        """Test if found solution is acceptable.

        - targeted evasion: check if score has increased
        - indiscriminate evasion: check if score has decreased

        Parameters
        ----------
        x0 : CArray
            Initial attack point
        y0 : CArray
            Label of the initial attack point.
        x_opt : CArray
            Final optimal point.
        clf : CClassifier
        y_target : int

        """
        init_pred, init_score = clf.predict(
            x0, return_decision_function=True)
        final_pred, final_score = clf.predict(
            x_opt, return_decision_function=True)

        if y_target is not None:
            self.assertGreater(final_score[:, y_target].item(),
                               init_score[:, y_target].item())

        self.assertLess(final_score[y0].item(), init_score[y0].item())
