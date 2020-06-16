from secml.testing import CUnitTest

from secml.core import CCreator


class Foo(CCreator):  # Toy class for testing
    def __init__(self):
        self.a = 1  # Public
        self._b = 2  # Read-only
        self._c = 3  # Read/Write
        self._d = 4  # Protected (no read or write)

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, val):
        self._c = val


class Doo(CCreator):  # Toy class for testing
    def __init__(self):
        self.a = 1  # Public


class Coo(CCreator):  # Toy class for testing
    def __init__(self):
        self._a = 1  # Protected (no read or write)


class TestCCreator(CUnitTest):
    """Unit test for secml.core.CCreator."""

    def setUp(self):
        self.test = Foo()

    def test_get_params(self):
        """Unittest for `CCreator.get_params()`."""

        def check_attrs(pdict, expected):
            self.assertTrue(set(attr for attr in pdict) == expected)

        # Standard class
        params = self.test.get_params()
        self.logger.info("Foo.get_params(): {:}".format(params))
        check_attrs(params, {'a', 'c'})

        # To the read-only param we assign a class with a public attribute
        # get_params should return the read-only parameter too
        self.test._b = Doo()
        params = self.test.get_params()
        self.logger.info("Foo.get_params() with b=Doo(): {:}".format(params))
        check_attrs(params, {'a', 'b', 'c'})

        # To the read-only param we assign a class with a protected attribute
        # get_params should NOT return the read-only parameter
        self.test._b = Coo()
        params = self.test.get_params()
        self.logger.info("Foo.get_params() with b=Coo(): {:}".format(params))
        check_attrs(params, {'a', 'c'})

        # In the following we replace the protected attribute
        # get_params should NOT return it in any case

        self.test._d = Doo()
        params = self.test.get_params()
        self.logger.info("Foo.get_params() with d=Doo(): {:}".format(params))
        check_attrs(params, {'a', 'c'})

        self.test._d = Coo()
        params = self.test.get_params()
        self.logger.info("Foo.get_params() with d=Coo(): {:}".format(params))
        check_attrs(params, {'a', 'c'})

    def test_set(self):
        """Unittest for `CCreator.set()`."""

        # Standard class
        self.test.set('a', 10)
        self.logger.info("Foo.a: {:}".format(self.test.a))
        self.assertEqual(self.test.a, 10)
        self.test.set('c', 30)
        self.assertEqual(self.test.c, 30)
        self.logger.info("Foo.a: {:}".format(self.test.c))

        with self.assertRaises(AttributeError):
            self.test.set('b', 20)
        with self.assertRaises(AttributeError):
            self.test.set('d', 40)

        # To the read-only param we assign a class with a public attribute
        # get_params should return the read-only parameter too
        self.test._b = Doo()
        self.test.set('b.a', 10)
        self.assertEqual(self.test.b.a, 10)
        self.logger.info("Foo.b: {:}".format(self.test.b))
        with self.assertRaises(AttributeError):
            self.test.set('b', 20)

        # To the read-only param we assign a class with a protected attribute
        # get_params should NOT return the read-only parameter
        self.test._b = Coo()
        with self.assertRaises(AttributeError):
            self.test.set('b.a', 10)
        with self.assertRaises(AttributeError):
            self.test.set('b', 20)


if __name__ == '__main__':
    CUnitTest.main()
