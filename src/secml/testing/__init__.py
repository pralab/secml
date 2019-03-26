try:
    import pytest
except ImportError:
    raise ImportError(
        "Install extra component `unittests` to use `secml.testing`")

from .c_unittest import CUnitTest
