# this program basically does a constexpr and generates cuda code
import textwrap
import numpy as np
from code_fragments import *
from utils import *
from ptx_utils import *
import random
import os

import argparse
parser = argparse.ArgumentParser(description='CodeGen V1')

parser.add_argument('--A_dim', type=int, default=12)
parser.add_argument('--B_dim', type=int, default=12)
parser.add_argument('--C_dim', type=int, default=12)
parser.add_argument('--A_blocks', type=int, default=12)
parser.add_argument('--C_blocks', type=int, default=12)
parser.add_argument('--Gy', type=int, default=12)
parser.add_argument('--infile', default=None, type=str)
parser.add_argument('--infile_bias', default=None, type=str)
parser.add_argument('--outfile', default=None, type=str)
parser.add_argument('--half',default=False,action='store_true')
parser.add_argument('--fuse',default=False,action='store_true')
parser.add_argument('--residual',default=False,action='store_true')
parser.add_argument('--no_relu',default=False,action='store_true')


args = parser.parse_args()
HALF = args.half
FUSE_END = args.fuse
RESIDUAL = args.residual
NO_RELU = args.no_relu
GY = args.Gy
A_dim = args.A_dim
B_dim = args.B_dim
C_dim = args.C_dim
A_blocks = args.A_blocks
C_blocks = args.C_blocks
input_file = args.infile
input_file_bias = args.infile_bias
outfile = args.outfile
BA = np.load(input_file)
if input_file_bias:
    bias = np.load(input_file_bias)
print(BA.shape)
BA = BA.squeeze()

if FUSE_END and GY > 1:
    print("More than one group not supported with epilogue strategy")
    exit()

print("WARNING! PTX MODE CURRENTLY ASSUMES FX = 1")

LOAD_CACHE = """
RC[ITER][i] = BC[IDX + C_offset + lane];
"""

LOAD_CACHE_PTX = """
ld.global.nc.f32    load_reg, [ADDR + OFFSET];"""

LOAD_CACHE_PTX_HALF = """
ld.global.nc.f16x2    load_reg, [ADDR + OFFSET];"""

MAIN_PROGRAM_PTX = """
fma.rn.f32  DEST, load_reg, MULT, ACC;"""

MAIN_PROGRAM_PTX_HALF = """
mov.u32 temp_reg, MULT;
fma.rn.f16x2  DEST, load_reg, temp_reg, ACC;"""

MAIN_PROGRAM_PTX_HALF_VIR = """
mov.u32 virg_reg, 0x00000000;
mov.u32 temp_reg, MULT;
fma.rn.f16x2  DEST, load_reg, temp_reg, virg_reg;"""




def emit_load_block_ptx(B_idx,block,ADDR):
    new_block = block.replace("ADDR",str(ADDR)).replace("OFFSET",str(B_idx * C_dim * 4))
    return new_block

def emit_compute_block_ptx(Ny_idx,block_reg_names,val,block,virgin):

    hex_number = float_to_hex(val)
    reg_name = "%f" + str(block_reg_names[Ny_idx])
    if virgin[Ny_idx] == 0:
        acc = "0f00000000"
        virgin[Ny_idx] = 1
    else:
        acc = reg_name
    return block.replace("DEST",reg_name).replace("ACC",acc).replace("MULT",hex_number)

def emit_compute_block_ptx_half(Ny_idx,block_reg_names,val,virgin):

    hex_number = half_to_hex(val)
    reg_name = "%f" + str(block_reg_names[Ny_idx])
    if virgin[Ny_idx] == 0:
        virgin[Ny_idx] = 1
        return MAIN_PROGRAM_PTX_HALF_VIR.replace("DEST",reg_name).replace("MULT",hex_number)
    else:
        acc = reg_name
        return MAIN_PROGRAM_PTX_HALF.replace("DEST",reg_name).replace("ACC",acc).replace("MULT",hex_number)

def ny_to_a(ny_idx,groupId,blockId, A_dim = None, A_offset = None):
    if A_offset is None:
        A_offset = blockId * (A_dim // A_blocks)
    return A_offset + ny_idx

def generate_cuda_stem(block,NY,GY = None):

    program = ""

    for group in range(GY):
        program += GROUP_CONTROL_START.replace("GROUP",str(group)) + "\n"

        program += textwrap.indent(GEN_LANDMARK_PTX.replace("I",str(block)).replace("J",str(group)),"\t")
        program += textwrap.indent(GEN_LOAD,"\t")

        for i in range(NY):
            program += textwrap.indent(GEN_ACC.replace("B",str(block)).replace("G",str(group)).replace("I",str(i)),"\t")

        program += textwrap.indent(GEN_END,"\t")
        program += GROUP_CONTROL_END + "\n"

    return program

def generate_from_B(Ny_indices, B_indices,BA,block,NY,reg_names, ADDR, GY = None,A_offset=None):

    ptxs = []

    for group in range(GY):

        if block == 0 and group == 0:
            if HALF:
                ptx = ".reg .f32 load_reg;\n\t.reg .f32 temp_reg;\n\t.reg .f32 virg_reg, bias_reg, pred_reg,zero_reg;\n\t mov.u32 zero_reg, 0x00000000;\n\t"
            else:
                ptx = ".reg .f32 load_reg;\n\t"
        else:
            ptx = "\n\t"

        block_reg_names = reg_names[block]
        virgin = np.zeros(NY)

        old_b_idx = -1

        for ny_idx, b_idx in zip(Ny_indices[group],B_indices[group]):

            if b_idx != old_b_idx:
                if HALF:
                    load_block_ptx = emit_load_block_ptx(b_idx,LOAD_CACHE_PTX_HALF,ADDR)
                else:
                    load_block_ptx = emit_load_block_ptx(b_idx,LOAD_CACHE_PTX,ADDR)
                ptx += load_block_ptx
                old_b_idx = b_idx

            a_idx = ny_to_a(ny_idx,group,block,A_dim = A_dim, A_offset=A_offset)
            value = BA[b_idx,a_idx]

            if HALF:
                compute_block_ptx = emit_compute_block_ptx_half(ny_idx, block_reg_names, value,  virgin)
            else:
                compute_block_ptx = emit_compute_block_ptx(ny_idx, block_reg_names, value, MAIN_PROGRAM_PTX, virgin)
            ptx += compute_block_ptx

        ptxs.append(textwrap.indent(ptx,"\t"))

    return ptxs


def get_idx_balanced(block,BA,A_offset,block_NY,GY=None):

    Ny_indices = [[] for i in range(GY)]
    B_indices = [[] for i in range(GY)]
    nnz = np.sum(np.abs(BA[:,A_offset:A_offset + block_NY]) > EPS )
    nnz_per_group = nnz // GY
    curr_group = 0
    curr_nnz = 0
    for B_idx in range(B_dim):
        for ny in range(block_NY):
            assert curr_group < GY
            A_idx = ny_to_a(ny,curr_group,block,A_dim = A_dim, A_offset=A_offset)
            if np.abs(BA[B_idx,A_idx]) > EPS:
                B_indices[curr_group].append(B_idx)
                Ny_indices[curr_group].append(ny)
                curr_nnz += 1
            if curr_nnz > nnz_per_group:
                curr_group += 1
                curr_nnz = 0

    return Ny_indices, B_indices

def load_balancer2(BA):

    total_nnz = (np.abs(BA) > EPS).sum()
    nnz_per_block = total_nnz / A_blocks
    sums = np.sum(np.abs(BA) > EPS, axis = 0)
    cs = np.cumsum(sums)
    bounds = [np.argmax(cs > nnz_per_block * i) for i in range(A_blocks)]
    bounds = bounds + [A_dim]
    nnzs = np.diff(bounds)
    NY = np.max(nnzs)
    return bounds, NY

def no_load_balance(BA):

    assert A_dim % A_blocks == 0
    interval = A_dim // A_blocks
    bounds = [interval * i for i in range(A_blocks + 1)]
    return bounds , interval

# name is the name of the numpy file
def gencode(BA,outfile,C_dim,A_blocks,C_blocks,GY,name=None):
    program = ""
    assert A_dim % A_blocks == 0
    assert C_dim % C_blocks == 0
    B_dim = BA.shape[0]
  #  bounds, NY = load_balancer2(BA)
    bounds, NY = no_load_balance(BA)
    if RESIDUAL:
        program += START_NONFUSED_RESIDUAL.replace("ST_VAL",str(ST)).replace("Ny",str(NY)).replace("GY",str(GY)).replace("A_dim",str(A_dim)).replace(
            "C_dim",str(C_dim)).replace("B_dim",str(B_dim)).replace("A_BLOCKS",str(A_blocks)).replace("C_BLOCKS",str(C_blocks)) + "\n"
    else:
        program += START_NONFUSED.replace("ST_VAL",str(ST)).replace("Ny",str(NY)).replace("GY",str(GY)).replace("A_dim",str(A_dim)).replace(
            "C_dim",str(C_dim)).replace("B_dim",str(B_dim)).replace("A_BLOCKS",str(A_blocks)).replace("C_BLOCKS",str(C_blocks)) + "\n"
    for block in range(A_blocks):
    #for block in range(1):

        A_offset = bounds[block]
        block_NY = bounds[block+1] - A_offset
        program += BLOCK_CONTROL_START.replace("BLOCK", str(block)) + "\n"
        program += textwrap.indent(generate_cuda_stem(block,NY,GY=GY),"\t") + "\n"
        if FUSE_END:
            if RESIDUAL:
                if NO_RELU:
                    print("already no relu, this is a mistake")
                    exit()
                my_block = BLOCK_END_REDUCTION_RESIDUAL
            else:
                my_block = BLOCK_END_REDUCTION

            for i in range(block_NY):
                program += my_block.replace("OFFSET",str((A_offset + i) * C_dim)).replace("IDX",str(i)).replace("BIAS",str(bias[A_offset+i]))
        else:
            if NO_RELU:
                print("Not supported")
                exit()
            if RESIDUAL:
                program += BLOCK_END_RESIDUAL.replace("A_offset",str(A_offset)).replace("Ny",str(block_NY)).replace("A_BLOCKS",str(A_blocks)).replace(
                "C_BLOCKS", str(C_blocks)).replace("A_dim",str(A_dim)).replace("C_dim",str(C_dim)).replace("B_dim",str(B_dim)) + "\n"
            else:
                program += BLOCK_END.replace("A_offset",str(A_offset)).replace("Ny",str(block_NY)).replace("A_BLOCKS",str(A_blocks)).replace(
                    "C_BLOCKS", str(C_blocks)).replace("A_dim",str(A_dim)).replace("C_dim",str(C_dim)).replace("B_dim",str(B_dim)) + "\n"
        program += BLOCK_CONTROL_END

    program += END_NONFUSED.replace("A_BLOCKS",str(A_blocks)).replace("C_BLOCKS", str(C_blocks)).replace("A_dim",str(A_dim)). \
        replace("C_dim",str(C_dim)).replace("B_dim",str(B_dim)).replace("AB_sparse_tidy.npy",name)

    token = str(random.random())[2:5]
    temp_cu_file_name = "temp_stub" + token + ".cu"
    temp_ptx_file_name = "temp_stub" + token + ".ptx"

    open(temp_cu_file_name,"w").write(program)
    os.system("nvcc -arch=sm_75 -I /h/s29hao/sparsert/build/cnpy -L /h/s29hao/sparsert/build/cnpy/build -w -O3 -ptx -o " + temp_ptx_file_name + " " + temp_cu_file_name + " --std=c++11 --compiler-options=\"-fsingle-precision-constant\" -lcnpy -lz")
    os.sync()
    reg_names , addresses = parse_ptx(temp_ptx_file_name,A_blocks)
    #print(reg_names)
    ptxs = []
    for block in range(A_blocks):
    #for block in range(1):
        A_offset = bounds[block]
        block_NY = bounds[block+1] - A_offset
        Ny_indices, B_indices = get_idx_balanced(block,BA,A_offset,block_NY,GY=GY)

        block_ptxs = generate_from_B(Ny_indices,B_indices,BA,block,NY,reg_names,addresses[block], GY=GY,A_offset=A_offset)
        ptxs.append(block_ptxs)

    if RESIDUAL or NO_RELU:
        insert_ptx(temp_ptx_file_name, outfile, ptxs,False)
    else:
        insert_ptx(temp_ptx_file_name, outfile, ptxs)

#GX = 191



gencode(BA,outfile,C_dim,A_blocks,C_blocks,GY,name=input_file)
