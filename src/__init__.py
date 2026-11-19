"""Document Q&A Agent package."""

import os
import sys

# FAISS and PyTorch can each create large OpenMP pools; keep this process bounded
# by default before either native library is imported. Users may set it earlier.
os.environ.setdefault("OMP_NUM_THREADS", "4")
if sys.platform == "win32":
    # The Windows wheels bundle different OpenMP runtimes (PyTorch's Intel
    # runtime and FAISS's LLVM runtime). Their native work here is sequential.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

