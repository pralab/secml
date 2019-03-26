try:
    import cleverhans
except ImportError:
    try:  # Skip unittests if cleverhans is not available
        from secml.testing import CUnitTest
        CUnitTest.importskip("cleverhans")
    except ImportError:  # CUnitTest not available
        pass
    raise ImportError("CleverHans is not available!")
