import sys
sys.path.append("../.")
from ortools.sat.python import cp_model
import numpy as np
import sympy as sp
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
from math import log2
from transition import quasi_differential_transition_matrix
from collections import Counter


ARADI_sbox = [0, 1, 2, 3, 4, 13, 15, 6, 8, 11, 5, 14, 12, 7, 10, 9]

def print_aradi_round(diff):
    c = 0
    for v in range(128):
        if((diff>>(127-v))&0x01==1):
            print("x", end=" ")
        else:
            print("-", end=" ")
        c+=1
        if(c==32):
            print("")
            c=0
    print(" ")

def print_masks(masks, solver):
    for vs in masks:
        c = 0
        for v in vs:
            c <<= 1
            c += solver.Value(v)
        print(hex(c)[2:].zfill(len(vs)//4),end="")

def value_masks(masks, solver):
    c = 0
    for vs in masks:
        for v in vs:
            c <<= 1
            c += solver.Value(v)
    return c

def print_correlation(prob,key_masks):
    #TODO: try to remove?
    #prob = "2^{-24}"
    #key_masks = ["0x40000000400000000000000000000000",...]

    for j in range(len(key_masks)):
        key_mask = key_masks[j]
        binary_str = bin(int(key_mask, 16))[2:].zfill(128)
        if int(key_mask, 16)!=0:
            indices = [i for i, bit in enumerate(binary_str) if bit == '1'] # indice i is the i-th bit with the 0-th bit the most significative one

            exponent = " + ".join(f"k^{j}_{{{i}}}" for i in indices)

            print(f"(-1)^{{{exponent}}}",end=" ")

    print(f"{prob}")

def print_correlation_subkeys(prob,key_masks,r):
    #prob = "2^{-24}"
    #key_masks = ["0x40000000400000000000000000000000",...]

    active_bits = [i for i in range((r+1)*128,-1,-1) if ((key_masks >> i) & 1) == 1]

    exponent = " + ".join(f"k^{r-i//128}_{{{127-i%128}}}" for i in active_bits)
    print(f"(-1)^{{{exponent}}}",end=" ")

    print(f"{prob}") 

D = quasi_differential_transition_matrix(lambda x: ARADI_sbox[x], 4, 4)[::16, ::16]
ARADI_differential_sbox_conditions = []
for i in range(16):
    for j in range(16):
        p = D[i, j]
        if p != 0:
            ARADI_differential_sbox_conditions.append(tuple((j>>k) & 1 for k in range(4)) + tuple((i>>k) & 1 for k in range(4)) + (-int(log2(abs(p))), ))

def aradi_differential_trails_round(model, input_vars, output_vars, round):
     # variables to track correlation
    cor_vars = [model.NewIntVar(0, 3, "") for _ in range(32)]
    
    # the shift variables are defined on the round
    shifts = [[11,8,14],[10,9,11],[9,4,14],[8,9,7]]
    a = shifts[round%4][0]
    b = shifts[round%4][1]
    c = shifts[round%4][2]

    sb_output_diff = [model.NewBoolVar("") for _ in range(128)]

    # application of the sbox layer
    for i in range(32):
        iv_SB = [input_vars[j] for j in range(127-i,-1,-32)]
        ov_SB = [sb_output_diff[j] for j in range(127-i,-1,-32)]
        model.AddAllowedAssignments(iv_SB + ov_SB + cor_vars[i:i+1], ARADI_differential_sbox_conditions)

    #application of the linear layer
    for i in range(4):
        iv_u = sb_output_diff[32*i: 32*i+16]
        iv_l = sb_output_diff[32*i+16: 32*i+32]
        ov_u = output_vars[32*i: 32*i+16]
        ov_l = output_vars[32*i+16: 32*i+32]
        for j in range(16):
            model.AddBoolXOr([iv_u[(j+a)%16],ov_u[j],iv_u[j],iv_l[(j+c)%16],1]) # the xor for u
            model.AddBoolXOr([iv_l[(j+a)%16],ov_l[j],iv_l[j],iv_u[(j+b)%16],1]) # the xor for l
            
    # return model and correlation variables
    return model, cor_vars

class TrailCollector3(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, trail_vars, corr_eq):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.trail_vars = trail_vars
        self.corr_eq = corr_eq
        self.res = {}

    def print_trails(self):
        probabilities = "probabilities = "
        print("chars = [",end="")
        for t, (c) in sorted(self.res.items(), key=lambda x: x[1]):
            #print(f"Trail with probability 2^{-c}.")
            probabilities += f",2^{-c}"
            print("[",end="")
            for s in t:
                if(s!=t[len(t)-1]):
                    print("0x",hex(s)[2:].zfill(len(self.trail_vars[0])//4),sep="",end=",")
                else:
                    print("0x",hex(s)[2:].zfill(len(self.trail_vars[0])//4),sep="",end="")
            print("],")
        print(probabilities)

    def number_of_trails(self):
        return len(self.res)

    def on_solution_callback(self):
        res = []
        for vs in self.trail_vars:
            c = 0
            for v in vs:
                c <<= 1
                c += self.Value(v)
            res.append(c)
        self.res[tuple(res)] = (self.Value(self.corr_eq))

QD = quasi_differential_transition_matrix(lambda x: ARADI_sbox[x], 4, 4)
ARADI_quasi_differential_sbox_conditions = []
for i in range(256):
    for j in range(256):
        c = QD[i, j]
        if c != 0:
            ARADI_quasi_differential_sbox_conditions.append(tuple((j>>k) & 1 for k in range(8)) + tuple((i>>k) & 1 for k in range(8)) + (-int(log2(abs(c))), 0 if c > 0 else 1))

def aradi_quasidifferential_trails_round_layer(model, input_mask, input_diff, output_mask, output_diff, round):
    # correlation and sign vars per S-box
    cor_vars = [model.NewIntVar(0, 3, "") for _ in range(32)]
    sign_vars = [model.NewBoolVar("") for _ in range(32)]

    # Temporary variables to hold the S-box layer output (no longer "external")
    sb_output_mask = [model.NewBoolVar("") for _ in range(128)]
    sb_output_diff = [model.NewBoolVar("") for _ in range(128)]

    # Apply S-box layer
    for i in range(32):
        id_SB = [input_diff[j] for j in range(127 - i, -1, -32)]
        im_SB = [input_mask[j] for j in range(127 - i, -1, -32)]
        od_SB = [sb_output_diff[j] for j in range(127 - i, -1, -32)]
        om_SB = [sb_output_mask[j] for j in range(127 - i, -1, -32)]
        model.AddAllowedAssignments(im_SB + id_SB + om_SB + od_SB + cor_vars[i:i + 1] + sign_vars[i:i + 1],ARADI_quasi_differential_sbox_conditions)

    # Apply linear layer to S-box outputs → output of round
    shifts = [[11, 8, 14], [10, 9, 11], [9, 4, 14], [8, 9, 7]]
    a = shifts[round % 4][0]
    b = shifts[round % 4][1]
    c = shifts[round % 4][2]

    for i in range(4):
        sb_id_u = sb_output_diff[32*i: 32*i+16]
        sb_id_l = sb_output_diff[32*i+16: 32*i+32]
        od_u = output_diff[32*i: 32*i+16]
        od_l = output_diff[32*i+16: 32*i+32]
        sb_im_u = sb_output_mask[32*i: 32*i+16]
        sb_im_l = sb_output_mask[32*i+16: 32*i+32]
        om_u = output_mask[32*i: 32*i+16]
        om_l = output_mask[32*i+16: 32*i+32]

        for j in range(16):
            # mask equations (inverse because of how they’re defined in linear layer)
            model.AddBoolXOr([sb_im_u[j], om_u[j], om_u[(j - a) % 16], om_l[(j - b) % 16], 1])
            model.AddBoolXOr([sb_im_l[j], om_l[j], om_l[(j - a) % 16], om_u[(j - c) % 16], 1])
            # diff equations
            model.AddBoolXOr([sb_id_u[(j + a) % 16], od_u[j], sb_id_u[j], sb_id_l[(j + c) % 16], 1])
            model.AddBoolXOr([sb_id_l[(j + a) % 16], od_l[j], sb_id_l[j], sb_id_u[(j + b) % 16], 1])

    return model, cor_vars, sign_vars

class TrailCollector4(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, trail_vars, corr_eq, sign_eq):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.trail_vars = trail_vars
        self.corr_eq = corr_eq
        self.sign_eq = sign_eq
        self.res = {}

    def print_trails(self):
        for t, (c, s) in sorted(self.res.items(), key=lambda x: x[1][0]):
            if (-1)**s==1:
                prob = f"2^{-c}"
            else:
                prob = f"-2^{-c}"
            hex_values = []
            for s in t:
                hex_values.append(f"0x{hex(s)[2:].zfill(len(self.trail_vars[0])//4)}")
            print_correlation(prob,hex_values)

    def print_masks(self):
        for t, (c, s) in sorted(self.res.items(), key=lambda x: x[1][0]):
            for qt in t:
                if(s!=t[len(t)-1]):
                    print(hex(qt)[2:].zfill(len(self.trail_vars[0])//4),sep="",end="")
                else:
                    print(hex(qt)[2:].zfill(len(self.trail_vars[0])//4),sep="",end="")
            print("-")
            

    def number_of_trails(self):
        return len(self.res)

    def on_solution_callback(self):
        res = []
        for vs in self.trail_vars:
            c = 0
            for v in vs:
                c <<= 1
                c += self.Value(v)
            res.append(c)
        self.res[tuple(res)] = (self.Value(self.corr_eq), self.Value(self.sign_eq) % 2)

def constraint_quotient_space(base, correlations, masks, model, r):
    z = 128 * (r + 1) + 1  # +1 for correlation bit per element
    bitvectors = []

    for b, corr in zip(base, correlations):
        bits = [int(bit) for bit in bin(b)[2:].zfill(z - 1)[- (z - 1):]]
        bits = bits + [corr]  # correlation bit = least significant bit
        bitvectors.append(bits)

    B = sp.Matrix(bitvectors)

    rref_matrix, pivot_columns = B.rref()
    for pivot in pivot_columns:
        if pivot < 128 * (r + 1):  # exclude correlation bits
            model.add(masks[pivot // 128][pivot % 128] == 0)

    return model

def hex_to_bin_vec2(hex, bitlength):  # Adjust bitlength as needed
    return np.array([int(b) for b in bin(hex)[2:].zfill(bitlength)], dtype=np.uint8)

def gf2_rref(A):
    A = A.copy() % 2
    m, n = A.shape
    pivots = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivot_rows = np.where(A[row:, col])[0]
        if pivot_rows.size == 0:
            continue
        i = pivot_rows[0] + row
        A[[row, i]] = A[[i, row]]  # swap rows
        for r in range(m):
            if r != row and A[r, col]:
                A[r] ^= A[row]  # XOR for GF(2)
        pivots.append(col)
        row += 1
    return A, pivots   # <- MUST return both!

def row_echelon_form(base, correlations, r):
    z = 128 * (r + 1) + 1  # +1 for correlation bit per element
    bitvectors = []

    for b, corr in zip(base, correlations):
        bits = hex_to_bin_vec2(b, z-1)               
        bits = np.append(bits, corr).astype(np.uint8)   
        bitvectors.append(bits)

    bitvectors = np.array(bitvectors, dtype=np.uint8)
    
    return gf2_rref(bitvectors)

def compress_table(table):
    keys = [tuple(row) for row in table]
    counts = Counter(keys)
    out = []
    seen = set()
    for key in keys:               
        if key in seen:
            continue
        seen.add(key)
        out.append(list(key) + [counts[key]])

    return out