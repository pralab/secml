from secml.testing import CUnitTest

from urllib import parse
import tempfile
import requests_mock

from secml.utils.download_utils import dl_file, dl_file_gitlab, md5
from secml.utils import fm


class TestDownloadUtils(CUnitTest):
    """Unittests for `utils.download_utils`."""

    def setUp(self):
        # Retrieve the temporary files directory
        self.tempdir = tempfile.gettempdir()
        # Url for the mock requests
        self.test_url = 'mock://test.utils.download_utils'

    @requests_mock.Mocker()
    def test_dlfile(self, m):
        """Test for `dl_file` standard beahavior."""

        # Test for an available text file
        url = self.test_url + '/test.txt'
        file_content = 'resp'

        m.get(url, text=file_content)
        out_file = dl_file(url, self.tempdir)

        with open(out_file) as f:
            self.assertEqual(file_content, f.read())

        # Test for an available text file with parameters in the url
        url = self.test_url + '/test.txt?id=1&out=45'
        file_content = 'resp'

        m.get(url, text=file_content)
        out_file = dl_file(url, self.tempdir)

        # Check if parameters have been correctly removed
        self.assertEqual('test.txt', fm.split(out_file)[1])

        with open(out_file) as f:
            self.assertEqual(file_content, f.read())

        # Test for an unavailable text file
        url = self.test_url + '/test2.txt'
        m.get(url, text='Not Found', status_code=404)
        with self.assertRaises(RuntimeError) as e:
            dl_file(url, self.tempdir)

        self.assertTrue(
            'File is not available (error code 404)' in str(e.exception))

    @requests_mock.Mocker()
    def test_dlfile_content_length(self, m):
        """Test for `dl_file` beahavior with 'content-length' header."""

        # Test for an available text file (with 'content-length' header)
        url = self.test_url + '/test.txt'
        file_content = 'resp'

        m.get(url, text=file_content, headers={'Content-Length': '4'})
        out_file = dl_file(url, self.tempdir)

        with open(out_file) as f:
            self.assertEqual(file_content, f.read())

    @requests_mock.Mocker()
    def test_dlfile_headers(self, m):
        """Test for `dl_file` beahavior with additional headers."""

        # Test for an available text file (with 'content-length' header)
        url = self.test_url + '/test.txt'
        file_content = 'resp'

        m.get(url, text=file_content, request_headers={'TOKEN': 'test'})

        out_file = dl_file(url, self.tempdir, headers={'TOKEN': 'test'})

        with open(out_file) as f:
            self.assertEqual(file_content, f.read())

        # Additional headers should be ignored
        out_file = dl_file(url, self.tempdir,
                           headers={'TOKEN': 'test', 'HEADER2': '2'})

        with open(out_file) as f:
            self.assertEqual(file_content, f.read())

        # download should fail if no header or wrong header is defined
        with self.assertRaises(Exception):
            dl_file(url, self.tempdir)
        with self.assertRaises(Exception):
            dl_file(url, self.tempdir, headers={'TOKEN': '2'})
        with self.assertRaises(Exception):
            dl_file(url, self.tempdir, headers={'HEADER2': 'test'})

    @requests_mock.Mocker()
    def test_dlfile_content_disposition(self, m):
        """Test for `dl_file` beahavior with 'Content-Disposition' header."""

        def _test_dlfile(cont_disp, fn='test.txt'):
            """

            Parameters
            ----------
            cont_disp : str
                Content of the 'Content-Disposition' header
            fn : str, optional
                Expected filename. Default 'text.txt'.

            """
            url = self.test_url + '/test.txt'
            file_content = 'resp'

            m.get(url, text=file_content,
                  headers={'Content-Length': '4',
                           'Content-Disposition': cont_disp})
            out_file = dl_file(url, self.tempdir)

            self.assertEqual(fn, fm.split(out_file)[1])

            with open(out_file) as f:
                self.assertEqual(file_content, f.read())

        # Test for text file (with 'content-disposition' header)
        disp = r'inline; filename="test.txt"; filename*=UTF-8\'\'test.txt'

        _test_dlfile(disp)

        # Check for 'content-disposition' filename different from url
        disp = r'inline; filename="READ.md"; filename*=UTF-8\'\'READ.md'

        _test_dlfile(disp, fn='READ.md')

        # Check for a simpler 'content-disposition' content
        disp = 'filename="READ.md"'

        _test_dlfile(disp, fn='READ.md')

        # Check for 'content-disposition' without filename
        disp = 'inline; test="test"'

        _test_dlfile(disp)

    def test_dl_file_md5(self):

        # Fixed long string to write to the file
        x = b'abcd' * 10000

        # Expected digest of the file
        md5_test = '3f0f597c3c69ce42e554fdad3adcbeea'

        # Generate a temp file to test
        with tempfile.NamedTemporaryFile(mode='wb') as fp:

            fp.write(x)

            md5_digest = md5(fp.name)

            self.logger.info("MD5: {:}".format(md5_digest))
            self.assertEqual(md5_test, md5_digest)

    @requests_mock.Mocker()
    def test_dlfile_gitlab(self, m):
        """Test for `dl_file_gitlab` standard beahavior."""

        repo = 'secml/test'
        file = 'files/test.txt'
        branch = 'master'

        api_url = 'https://gitlab.com/api/v4/projects/' \
                  '{:}/repository/files/{:}/raw?ref={:}'

        url = api_url.format(
            parse.quote(repo, safe=''),
            parse.quote(file, safe=''),
            branch)
        file_content = 'resp'

        # Mimic the response given by GitLab API
        disp = r'inline; filename="test.txt"; filename*=UTF-8\'\'test.txt'
        m.get(url, text=file_content, headers={'Content-Length': '4',
                                               'Content-Disposition': disp})

        out_file = dl_file_gitlab(repo, file, self.tempdir, branch=branch)

        with open(out_file) as f:
            self.assertEqual(file_content, f.read())

        # Testing multiple similar values for repo and file parameters
        dl_file_gitlab(
            repo + '/', file, self.tempdir, branch=branch)
        dl_file_gitlab(
            'gitlab.com/' + repo, file, self.tempdir, branch=branch)
        dl_file_gitlab(
            'gitlab.com/' + repo + '/', file, self.tempdir, branch=branch)
        dl_file_gitlab(
            'https://gitlab.com/' + repo, file, self.tempdir, branch=branch)
        dl_file_gitlab(
            'http://gitlab.com/' + repo, file, self.tempdir, branch=branch)
        dl_file_gitlab(
            repo, '/' + file, self.tempdir, branch=branch)

        # Testing wrong inputs
        with self.assertRaises(requests_mock.NoMockAddress):
            dl_file_gitlab(repo, file, self.tempdir, branch='develop')

        with self.assertRaises(requests_mock.NoMockAddress):
            dl_file_gitlab(repo, 'test.txt', self.tempdir, branch=branch)

        with self.assertRaises(requests_mock.NoMockAddress):
            dl_file_gitlab('secml/secml', file, self.tempdir, branch=branch)


if __name__ == '__main__':
    CUnitTest.main()
