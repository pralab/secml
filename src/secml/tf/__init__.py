try:
    import tensorflow
except ImportError:
    try:  # Skip unittests if tensorflow is not available
        from secml.utils import CUnitTest
        CUnitTest.importskip("tensorflow")
    except ImportError:  # CUnitTest not available
        pass
    raise ImportError("tensorflow is not available!")
