from secml.array.tests import CArrayTestCases

from secml.utils import fm
from secml.array import CArray


class TestCArraySaveLoad(CArrayTestCases):
    """Unit test for CArray SAVE/LOAD methods."""

    def test_save_load(self):
        """Test save/load of CArray"""
        self.logger.info("UNITTEST - CArray - save/load")

        test_file = fm.join(fm.abspath(__file__), 'test.txt')
        test_file_2 = fm.join(fm.abspath(__file__), 'test2.txt')

        # Cleaning test files
        try:
            fm.remove_file(test_file)
            fm.remove_file(test_file_2)
        except (OSError, IOError) as e:
            if e.errno != 2:
                raise e

        self.logger.info(
            "UNITTEST - CArray - Testing save/load for sparse matrix")

        self.array_sparse.save(test_file)

        # Saving to a file handle is not supported for sparse arrays
        with self.assertRaises(NotImplementedError):
            with open(test_file_2, 'w') as f:
                self.array_sparse.save(f)

        loaded_array_sparse = CArray.load(
            test_file, arrayformat='sparse', dtype=int)

        self.assertFalse((loaded_array_sparse != self.array_sparse).any(),
                         "Saved and loaded arrays (sparse) are not equal!")

        self.logger.info(
            "UNITTEST - CSparse - Testing save/load for dense matrix")

        self.array_dense.save(test_file, overwrite=True)

        loaded_array_dense = CArray.load(test_file, arrayformat='dense', dtype=int)

        self.assertFalse((loaded_array_dense != self.array_dense).any(),
                         "Saved and loaded arrays (sparse) are not equal!")

        # Checking sparse/dense equality between loaded data
        self.assertFalse((loaded_array_sparse.todense() != loaded_array_dense).any(),
                         "Loaded arrays are not equal!")

        # Only 'dense' and 'sparse' arrayformat are supported
        with self.assertRaises(ValueError):
            CArray.load(test_file, arrayformat='test')

        # Cleaning test files
        try:
            fm.remove_file(test_file)
            fm.remove_file(test_file_2)
        except (OSError, IOError) as e:
            if e.errno != 2:
                raise e


if __name__ == '__main__':
    CArrayTestCases.main()
