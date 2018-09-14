from c_kernel import CKernel

# Load optimized kernels if NUMBA is available
from secml.core.settings import USE_NUMBA
if USE_NUMBA is True:
    try:
        import numba
    except ImportError:
        import warnings
        warnings.warn("`numba` library is not available, loading standard kernels.")
        # Import sklearn kernels (add new entries also after below else)
        from c_kernel_linear import CKernelLinear
        from c_kernel_rbf import CKernelRBF
        from c_kernel_poly import CKernelPoly
        from c_kernel_histintersect import CKernelHistIntersect
        from c_kernel_laplacian import CKernelLaplacian
    else:
        from numba_kernel.c_kernel_linear_numba import CKernelLinearNumba as CKernelLinear
        from numba_kernel.c_kernel_rbf_numba import CKernelRBFNumba as CKernelRBF
        from numba_kernel.c_kernel_poly_numba import CKernelPolyNumba as CKernelPoly
        from numba_kernel.c_kernel_histintersect_numba import CKernelHistIntersectNumba as CKernelHistIntersect
        from numba_kernel.c_kernel_laplacian_numba import CKernelLaplacianNumba as CKernelLaplacian
else:
    # Import sklearn kernels (add new entries also after upper except)
    from c_kernel_linear import CKernelLinear
    from c_kernel_rbf import CKernelRBF
    from c_kernel_poly import CKernelPoly
    from c_kernel_histintersect import CKernelHistIntersect
    from c_kernel_laplacian import CKernelLaplacian

# kernels without numba support
from c_kernel_hamming import CKernelHamming
from c_kernel_euclidean import CKernelEuclidean
from c_kernel_max import CKernelMax
