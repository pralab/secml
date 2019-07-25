from secml.testing import CUnitTest

import sys

from secml.utils.download_utils import md5


class TestDownloadUtils(CUnitTest):
    """Unittests for `secml.core`download_utils`."""

    def test_md5(self):

        # Fixed long string to write to the file
        x = b'abcd' * 10000

        # Expected digest of the file
        # TODO: cause of py3 vs py2 encoding differences
        #  the md5 changes. Workaround to be investigated
        if sys.version_info < (3, 0):
            md5_test = '934f48695f0286ccfa0dddc060e2f152'
        else:
            md5_test = '3f0f597c3c69ce42e554fdad3adcbeea'

        # Generate a temp file to test
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb') as fp:

            fp.write(x)

            md5_digest = md5(fp.name)

            self.logger.info("MD5: {:}".format(md5_digest))
            self.assertEqual(md5_test, md5_digest)


if __name__ == '__main__':
    CUnitTest.main()
