try:
    import torch
except ImportError:
    try:  # Skip unittests if torch is not available
        from secml.utils import CUnitTest
        CUnitTest.importskip("torch")
    except ImportError:  # CUnitTest not available
        pass
    raise ImportError("PyTorch is not available!")
