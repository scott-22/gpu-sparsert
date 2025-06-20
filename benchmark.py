#!/usr/bin/env python3
import argparse
import numpy as np
import subprocess
import sys

# Edit this to contain the paths to include directories (that aren't already in include path)
NVCC_INCLUDE_PATHS = ["-I /h/s29hao/sparsert/build/include"]

# Edit this to contain the paths to library directories (that aren't already in library path)
# Make sure to include them in `LD_LIBRARY_PATH` when running the benchmark
NVCC_LINK_PATHS = ["-L /h/s29hao/sparsert/build/lib", "-L /pkgs/cuda-11.0/lib64"]

# Also, edit the paths in `sparsednn/code_gen_ptx.py`

# We are doing the matrix multiplication C = A * B, where A is the weight matrix and B is the input matrix
# Suppose A has shape (M, K) and B has shape (K, N), then C has shape (M, N)

FORMAT = "NCHW"

parser = argparse.ArgumentParser(description="Benchmarking script for SpMM")
parser.add_argument("name", help="name of solution we are benchmarking ('cublas', 'sparsert')")
parser.add_argument("source", help="source of the benchmark data ('mobilenet', 'sparsednn-1024')")
parser.add_argument("index", type=int, default=0, help="index of the data we are benchmarking (0-12 for mobilenet, 1-1920 for sparsednn-1024)")
args = parser.parse_args()

def load_weight_matrix(path: str, N: int):
    """
    Load weight matrix from an npy file, and generate a random input matrix along with a reference output matrix.
    We pass in N as the number of columns of the input matrix.
    """
    wt_matrix = np.load(path)
    _, K = wt_matrix.shape
    in_matrix = np.random.normal(size=(K, N))
    out_matrix = np.dot(wt_matrix, in_matrix)
    np.save("A.npy", wt_matrix.astype(np.float32))
    np.save("B.npy", in_matrix.astype(np.float32))
    np.save("ref.npy", out_matrix.astype(np.float32))
    np.save("ref_transposed.npy", out_matrix.astype(np.float32).transpose())

if args.name not in ["cublas", "sparsert"]:
    print("Invalid solution name to benchmark")
    sys.exit(-1)

if args.source not in ["mobilenet", "sparsednn-1024"]:
    print("Invalid benchmark data source")
    sys.exit(-1)

if args.source == "mobilenet":
    if args.index < 0 or args.index > 12:
        print("Invalid index")

    # Use hardcoded values for dimensions M, N, K. The logic for these values (and the subsequent block allocation) is in autotune_float.sh
    # We skip the autotuning process, since the code currently in autotune_float.sh doesn't actually do any autotuning, so I'm not sure
    # what it should look like.
    DIMENSIONS = [(64, 32, 12544), (128, 64, 3136), (128, 128, 3136), (256, 128, 784), (256, 256, 784), (512, 256, 196), (512, 512, 196), (1024, 512, 49), (1024, 1024, 49)]
    M, K, N = DIMENSIONS[args.index]
    load_weight_matrix(f"mobilenet/contraction_1x1_{args.index}.npy", N)
    A_BLOCKS = M // 8    # Outdated name, should be "M_BLOCKS" but kept for backward compatibility
    C_BLOCKS = N // 49   # Same as above, should be "N_BLOCKS"
    Gy = 1

    if args.name == "sparsert":
        subprocess.run([
            "python", "sparsednn/code_gen_ptx.py", 
            "--A_dim", str(M), "--B_dim", str(K), "--C_dim", str(N),
            "--A_blocks", str(A_BLOCKS), "--C_blocks", str(C_BLOCKS), "--Gy", str(Gy),
            "--infile", f"mobilenet/contraction_1x1_{args.index}_transposed.npy",  # SparseRT seems to require transpose of weight matrix
            "--outfile", "testing.ptx"
        ], check=True)

        subprocess.run([
            "ptxas", "-arch=sm_75", "testing.ptx", "-o", "testing.cubin"
        ], check=True)

        subprocess.run(
            [
                "nvcc", "sparsednn/driver_spmm.cpp", "-w", "-O3",
                f"-DM_dim={M},K_dim={K},N_dim={N},A_Blocks={A_BLOCKS},C_Blocks={C_BLOCKS},Gy={Gy}",
                "-lcuda", "-lcudart", "-lcnpy", "-lcublas", "-o", "sparsert", "--std=c++11", "-Xptxas=-v"
            ] + NVCC_INCLUDE_PATHS + NVCC_LINK_PATHS,
            check=True,
        )
    elif args.name == "cublas":
        subprocess.run(
            [
                "nvcc", "sparsednn/driver_spmm.cpp", "-w", "-O3",
                f"-DM_dim={M},K_dim={K},N_dim={N},A_Blocks={A_BLOCKS},C_Blocks={C_BLOCKS},Gy={Gy},TESTCUBLAS=1",
                "-lcuda", "-lcudart", "-lcnpy", "-lcublas", "-o", "sparsert", "--std=c++11", "-Xptxas=-v"
            ] + NVCC_INCLUDE_PATHS + NVCC_LINK_PATHS,
            check=True,
        )

    subprocess.run(
        "./sparsert > runtime",
        shell=True,
        check=True,
    )

    equiv_output = subprocess.run(
        ["python", "scripts/test_equivalence.py", "ref.npy", "ptx_result.npy"],
        check=True,
        capture_output=True,
    )
    print(equiv_output.stdout)

    with open("runtime", "r") as f:
        runtime = f.read()
        print(runtime)

elif args.source == "sparsednn-1024":
    if args.index <= 0 or args.index > 1920:
        print("Invalid index")
else:
    print("Invalid source")