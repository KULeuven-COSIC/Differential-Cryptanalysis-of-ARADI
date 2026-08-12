import sympy as sp
from sympy import Matrix
from sympy import expand, Add

def simplify_mod2(expr):
    expr = expand(expr)
    if isinstance(expr, Add):
        return sum((c % 2) * t for t, c in expr.as_coefficients_dict().items())
    return expr % 2

def mod2_matrix(mat):
    return mat.applyfunc(simplify_mod2)

def round_index_vector(i):
    bin_str = f"{i:08b}" 
    return Matrix([0] * 248 + [int(b) for b in bin_str])

M0 = [ #(x,y)->x
    [0x40000000,0x80000000], [0x20000000,0x40000000], [0x10000000,0x20000000], [0x08000000,0x10000000],
    [0x04000000,0x08000000], [0x02000000,0x04000000], [0x01000000,0x02000000], [0x00800000,0x01000000],
    [0x00400000,0x00800000], [0x00200000,0x00400000], [0x00100000,0x00200000], [0x00080000,0x00100000],
    [0x00040000,0x00080000], [0x00020000,0x00040000], [0x00010000,0x00020000], [0x00008000,0x00010000],
    [0x00004000,0x00008000], [0x00002000,0x00004000], [0x00001000,0x00002000], [0x00000800,0x00001000],
    [0x00000400,0x00000800], [0x00000200,0x00000400], [0x00000100,0x00000200], [0x00000080,0x00000100],
    [0x00000040,0x00000080], [0x00000020,0x00000040], [0x00000010,0x00000020], [0x00000008,0x00000010],
    [0x00000004,0x00000008], [0x00000002,0x00000004], [0x00000001,0x00000002], [0x80000000,0x00000001],
    #(x,y)-> y
    [0x40000000,0x90000000], [0x20000000,0x48000000], [0x10000000,0x24000000], [0x08000000,0x12000000],
    [0x04000000,0x09000000], [0x02000000,0x04800000], [0x01000000,0x02400000], [0x00800000,0x01200000],
    [0x00400000,0x00900000], [0x00200000,0x00480000], [0x00100000,0x00240000], [0x00080000,0x00120000],
    [0x00040000,0x00090000], [0x00020000,0x00048000], [0x00010000,0x00024000], [0x00008000,0x00012000],
    [0x00004000,0x00009000], [0x00002000,0x00004800], [0x00001000,0x00002400], [0x00000800,0x00001200],
    [0x00000400,0x00000900], [0x00000200,0x00000480], [0x00000100,0x00000240], [0x00000080,0x00000120],
    [0x00000040,0x00000090], [0x00000020,0x00000048], [0x00000010,0x00000024], [0x00000008,0x00000012],
    [0x00000004,0x00000009], [0x00000002,0x80000004], [0x00000001,0x40000002], [0x80000000,0x20000001]
]

M1 = [#(x,y)->x
    [0x00400000,0x80000000], [0x00200000,0x40000000], [0x00100000,0x20000000], [0x00080000,0x10000000],
    [0x00040000,0x08000000], [0x00020000,0x04000000], [0x00010000,0x02000000], [0x00008000,0x01000000],
    [0x00004000,0x00800000], [0x00002000,0x00400000], [0x00001000,0x00200000], [0x00000800,0x00100000],
    [0x00000400,0x00080000], [0x00000200,0x00040000], [0x00000100,0x00020000], [0x00000080,0x00010000],
    [0x00000040,0x00008000], [0x00000020,0x00004000], [0x00000010,0x00002000], [0x00000008,0x00001000],
    [0x00000004,0x00000800], [0x00000002,0x00000400], [0x00000001,0x00000200], [0x80000000,0x00000100],
    [0x40000000,0x00000080], [0x20000000,0x00000040], [0x10000000,0x00000020], [0x08000000,0x00000010],
    [0x04000000,0x00000008], [0x02000000,0x00000004], [0x01000000,0x00000002], [0x00800000,0x00000001],
    #(x,y)->y
    [0x00400000,0x80000008], [0x00200000,0x40000004], [0x00100000,0x20000002], [0x00080000,0x10000001],
    [0x00040000,0x88000000], [0x00020000,0x44000000], [0x00010000,0x22000000], [0x00008000,0x11000000],
    [0x00004000,0x08800000], [0x00002000,0x04400000], [0x00001000,0x02200000], [0x00000800,0x01100000],
    [0x00000400,0x00880000], [0x00000200,0x00440000], [0x00000100,0x00220000], [0x00000080,0x00110000],
    [0x00000040,0x00088000], [0x00000020,0x00044000], [0x00000010,0x00022000], [0x00000008,0x00011000],
    [0x00000004,0x00008800], [0x00000002,0x00004400], [0x00000001,0x00002200], [0x80000000,0x00001100],
    [0x40000000,0x00000880], [0x20000000,0x00000440], [0x10000000,0x00000220], [0x08000000,0x00000110],
    [0x04000000,0x00000088], [0x02000000,0x00000044], [0x01000000,0x00000022], [0x00800000,0x00000011]
]

# Identity and zero matrices
I = sp.eye(32)
Z32 = sp.zeros(32, 32)
Z64 = sp.zeros(64, 64)

# Convert a 32-bit integer to a binary list
def int_to_binary_list(x, bit_width=32):
    return [(x >> i) & 1 for i in range(bit_width - 1, -1, -1)]

# Convert M0 and M1 to binary matrices
M0_bin = sp.Matrix([sum([int_to_binary_list(cell) for cell in row], []) for row in M0])
M1_bin = sp.Matrix([sum([int_to_binary_list(cell) for cell in row], []) for row in M1])

# Construct the S matrix
S = sp.BlockMatrix([
    [M0_bin, Z64, Z64, Z64],
    [Z64, M1_bin, Z64, Z64],
    [Z64, Z64, M0_bin, Z64],
    [Z64, Z64, Z64, M1_bin]
]).as_explicit()

# First linear layer (L1)
L1 = sp.BlockMatrix([
    [I, Z32, Z32, Z32, Z32, Z32, Z32, Z32],
    [Z32, Z32, I, Z32, Z32, Z32, Z32, Z32],
    [Z32, I, Z32, Z32, Z32, Z32, Z32, Z32],
    [Z32, Z32, Z32, I, Z32, Z32, Z32, Z32],
    [Z32, Z32, Z32, Z32, I, Z32, Z32, Z32],
    [Z32, Z32, Z32, Z32, Z32, Z32, I, Z32],
    [Z32, Z32, Z32, Z32, Z32, I, Z32, Z32],
    [Z32, Z32, Z32, Z32, Z32, Z32, Z32, I]
]).as_explicit()

# Second linear layer (L2)
L2 = sp.BlockMatrix([
    [I, Z32, Z32, Z32, Z32, Z32, Z32, Z32],
    [Z32, Z32, Z32, Z32, I, Z32, Z32, Z32],
    [Z32, Z32, I, Z32, Z32, Z32, Z32, Z32],
    [Z32, Z32, Z32, Z32, Z32, Z32, I, Z32],
    [Z32, I, Z32, Z32, Z32, Z32, Z32, Z32],
    [Z32, Z32, Z32, Z32, Z32, I, Z32, Z32],
    [Z32, Z32, Z32, I, Z32, Z32, Z32, Z32],
    [Z32, Z32, Z32, Z32, Z32, Z32, Z32, I]
]).as_explicit()



key_sym = sp.symbols('k_0:256')
key_bin = Matrix(key_sym)

K0 = key_bin
K1 = L1 * (S * key_bin)
K2 = mod2_matrix(L2 * (S * K1) + round_index_vector(1))
K3 = mod2_matrix(L1 * (S * K2) + round_index_vector(2))
K4 = mod2_matrix(L2 * (S * K3) + round_index_vector(3))
K5 = mod2_matrix(L1 * (S * K4) + round_index_vector(4))
K6 = mod2_matrix(L2 * (S * K5) + round_index_vector(5))
K7 = mod2_matrix(L1 * (S * K6) + round_index_vector(6))
K8 = mod2_matrix(L2 * (S * K7) + round_index_vector(7))
K9 = mod2_matrix(L1 * (S * K8) + round_index_vector(8))
K10 = mod2_matrix(L2 * (S * K9) + round_index_vector(9))
K11 = mod2_matrix(L1 * (S * K10) + round_index_vector(10))
K12 = mod2_matrix(L2 * (S * K11) + round_index_vector(11))
K13 = mod2_matrix(L1 * (S * K12) + round_index_vector(12))
K14 = mod2_matrix(L2 * (S * K13) + round_index_vector(13))
K15 = mod2_matrix(L1 * (S * K14) + round_index_vector(14))
K16 = mod2_matrix(L2 * (S * K15) + round_index_vector(15))

def print_correlation_masterkey(prob,key_masks,r):
    active_bits = [i for i in range((r+1)*128,-1,-1) if ((key_masks >> i) & 1) == 1]
    exponent = 0
    for j in active_bits:
        Kj = globals()[f"K{j//128}"]
        if (j//128)%2 ==0:
            exponent = simplify_mod2(exponent + Kj[127-j%128])
        else:
            exponent = simplify_mod2(exponent + Kj[255-j%128])
    
    print(f"(-1)^{{{exponent}}}",end=" ")
    print(f"{prob}")  