from secml.testing import CUnitTest

from urllib import parse
import requests_mock
import re

import secml
from secml.settings import SECML_MODELS_DIR
from secml.model_zoo import load_model
from secml.model_zoo.load_model import MODELS_DICT_PATH, _logger
from secml.utils import fm


class TestModelZoo(CUnitTest):
    """Unittests for `model_zoo`."""

    @classmethod
    def setUpClass(cls):

        CUnitTest.setUpClass()

        # We now prepare all the urls and path required to mock requests
        # via gitlab API to https://gitlab.com/secml/secml-zoo repository

        # Fake models definitions
        cls.test_models_def = \
            fm.join(fm.abspath(__file__), 'models_dict_test.json')

        # Test model's definition
        cls.test_model_id = '_test_model'
        cls.test_model = \
            fm.join(fm.abspath(__file__), '_test_model_clf.py')
        cls.test_model_state = \
            fm.join(fm.abspath(__file__), '_test_model-clf.gz')

        # Url for mocking requests to the model zoo repository
        repo = parse.quote('secml/secml-zoo', safe='')
        file_model = parse.quote('models/_test/_test_model_clf.py', safe='')
        file_state = parse.quote('models/_test/_test_model-clf.gz', safe='')
        file_defs = parse.quote('models_dict.json', safe='')
        vers = 'v' + re.search(r'^\d+.\d+', secml.__version__).group(0)

        api_url = 'https://gitlab.com/api/v4/projects/' \
                  '{:}/repository/files/{:}/raw?ref={:}'

        # One url for master branch, one for current library version
        # One for model file, one for state file
        cls.api_url_model_master = api_url.format(repo, file_model, 'master')
        cls.api_url_model_vers = api_url.format(repo, file_model, vers)
        cls.api_url_state_master = api_url.format(repo, file_state, 'master')
        cls.api_url_state_vers = api_url.format(repo, file_state, vers)
        cls.api_url_defs_master = api_url.format(repo, file_defs, 'master')
        cls.api_url_defs_vers = api_url.format(repo, file_defs, vers)

        cls.api_model_headers = {
            'Content-Disposition':  r'inline; filename="_test_model_clf.py"'}
        cls.api_state_headers = {
            'Content-Disposition': r'inline; filename="_test_model-clf.gz"'}
        cls.api_defs_headers = {
            'Content-Disposition': r'inline; filename="models_dict.json"'}

        # Set the debug level of models loader to debug
        _logger.set_level('DEBUG')

    def setUp(self):

        # Remove existing 'models_dict.json' before testing
        if fm.file_exist(MODELS_DICT_PATH):
            fm.remove_file(MODELS_DICT_PATH)

    def tearDown(self):

        # Remove existing 'models_dict.json' before testing
        if fm.file_exist(MODELS_DICT_PATH):
            fm.remove_file(MODELS_DICT_PATH)

        # Removing folder with test model (force 'cause not empty)
        if fm.folder_exist(fm.join(SECML_MODELS_DIR, '_test')):
            fm.remove_folder(fm.join(SECML_MODELS_DIR, '_test'), force=True)

    def _mock_requests(self, m, defs_url=None, model_url=None, state_url=None):
        """Mock model zoo resources download requests.

        Mocker will be edited in-place.

        If an url is not provided, corresponding response will not be mocked.

        Parameters
        ----------
        m : requests_mock.Mocker
        defs_url : str or None, optional
        model_url : str or None, optional
        state_url : str or None, optional

        """
        if defs_url:  # Mocking models definitions
            self.logger.info("Mocking `{:}`".format(defs_url))
            with open(self.test_models_def) as fdefs:
                m.get(defs_url,
                      text=fdefs.read(),
                      headers=self.api_defs_headers)

        if model_url:  # Mocking model
            self.logger.info("Mocking `{:}`".format(model_url))
            with open(self.test_model) as fmodel:
                m.get(model_url,
                      text=fmodel.read(),
                      headers=self.api_model_headers)

        if state_url:  # Mocking model state
            self.logger.info("Mocking `{:}`".format(state_url))
            with open(self.test_model_state, 'br') as fmodelstate:
                # We read the state file as binary data and pass to content
                m.get(state_url,
                      content=fmodelstate.read(),
                      headers=self.api_state_headers)

    def _check_test_model(self):
        """Load the test model and check its parameters."""
        clf = load_model(self.test_model_id)
        self.logger.info("Loaded test model: {:}".format(clf))
        # The loaded test model should have "C=100"
        self.assertEqual(100, clf.C)

    def _test_load_model(self, defs_url, model_url, state_url):
        """Test for `load_model` valid behavior.

        We test the following:
         - all valid requests
         - a need for updating models dict and redownload model
         - a need for updating models dict and redownload model
           with a connection error when download models dict

        Parameters
        ----------
        defs_url : str or None, optional
        model_url : str or None, optional
        state_url : str or None, optional

        """
        with requests_mock.Mocker() as m:

            # Simulate a fine process, with all resources available
            self._mock_requests(
                m, defs_url=defs_url, model_url=model_url, state_url=state_url)

            self._check_test_model()  # Call model loading

            # We now simulate a need for `models_dict.json` update
            # by removing `.last_update` file
            fm.remove_file(fm.join(SECML_MODELS_DIR, '.last_update'))
            # Also remove test model to force re-download
            fm.remove_folder(fm.join(SECML_MODELS_DIR, '_test'), force=True)

            self._check_test_model()  # Call model loading

        # We now simulate a need for `models_dict.json` update,
        # but a connection error occurs (simulated by not mocking dl url)
        # Last available version of models dict should be used
        fm.remove_file(fm.join(SECML_MODELS_DIR, '.last_update'))
        fm.remove_folder(fm.join(SECML_MODELS_DIR, '_test'), force=True)

        with requests_mock.Mocker() as m:
            # Do not mock the url for models definitions
            self._mock_requests(
                m, defs_url=None, model_url=model_url, state_url=state_url)

            self._check_test_model()  # Call model loading

    def test_load_model_vers(self):
        """Test for `load_model` standard behavior (dl from version branch)."""

        self._test_load_model(defs_url=self.api_url_defs_vers,
                              model_url=self.api_url_model_vers,
                              state_url=self.api_url_state_vers)

    def test_load_model_master(self):
        """Test for `load_model` standard behavior (dl from master branch)."""

        self._test_load_model(defs_url=self.api_url_defs_master,
                              model_url=self.api_url_model_master,
                              state_url=self.api_url_state_master)

    def test_load_model_fail(self):
        """Test for `load_model` fail behavior."""
        # Cannot download `models_dict.json`
        with self.assertRaises(requests_mock.NoMockAddress):
            with requests_mock.Mocker():  # To intercept real http requests
                self._check_test_model()

        # Models defs can be download, but not the model
        with self.assertRaises(requests_mock.NoMockAddress):
            with requests_mock.Mocker() as m:
                self._mock_requests(
                    m, defs_url=self.api_url_defs_vers,
                    model_url=None, state_url=self.api_url_state_vers)
                self._check_test_model()

        # Models defs can be download, but not the model state
        with self.assertRaises(requests_mock.NoMockAddress):
            with requests_mock.Mocker() as m:
                self._mock_requests(
                    m, defs_url=self.api_url_defs_vers,
                    model_url=self.api_url_model_vers, state_url=None)
                self._check_test_model()

        # Can download defs, but requested model not available
        with self.assertRaises(KeyError):
            with requests_mock.Mocker() as m:
                self._mock_requests(
                    m, defs_url=self.api_url_defs_vers,
                    model_url=None, state_url=None)
                load_model('svm-test')


if __name__ == '__main__':
    CUnitTest.main()
