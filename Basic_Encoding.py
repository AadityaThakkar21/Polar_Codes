import numpy as np
import math as m

reliability_seq = [0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 128, 12, 33, 65, 20,
    256, 34, 24, 36, 7, 129, 66, 512, 11, 40, 68, 130, 19, 13, 48, 14, 72, 257, 21, 132,
    35, 258, 26, 513, 80, 37, 25, 22, 136, 260, 264, 38, 514, 96, 67, 41, 144, 28, 69, 42,
    516, 49, 74, 272, 160, 520, 288, 528, 192, 544, 70, 44, 131, 81, 50, 73, 15, 320, 133, 52, 
    23, 134, 384, 76, 137, 82, 56, 27, 97, 39, 259, 84, 138, 145, 261, 29, 43, 98, 515, 88, 
    140, 30, 146, 71, 262, 265, 161, 576, 45, 100, 640, 51, 148, 46, 75, 266, 273, 517, 104, 162,  
    53, 193, 152, 77, 164, 768, 268, 274, 518, 54, 83, 57, 521, 112, 135, 78, 289, 194, 85, 276, 
    522, 58, 168, 139, 99, 86, 60, 280, 89, 290, 529, 524, 196, 141, 101, 147, 176, 142, 530, 321,  
    31, 200, 90, 545, 292, 322, 532, 263, 149, 102, 105, 304, 296, 163, 92, 47, 267, 385, 546, 324
]

def insert_message(n, k, msg, rel_seq):
    u = np.zeros(n, dtype=int)
    start = n - k
    for i in range(k):
        u[rel_seq[start + i]] = msg[i]
    return u

n_bits = 32
k_bits = 20
levels = int(m.log2(n_bits))

rs_n = np.array([i for i in reliability_seq if i < n_bits])
print(f"Reliability sequence (N={n_bits}):", rs_n)

msg_bits = np.random.randint(0, 2, k_bits, dtype=int)
print("Message bits:", msg_bits)

u_seq = insert_message(n_bits, k_bits, msg_bits, rs_n)
print("U sequence:", u_seq)

def encode_iter(u, n):
    x = np.copy(u)
    m = 1
    for _ in range(levels - 1, -1, -1):
        for i in range(0, n, 2 * m):
            left = x[i:i + m]
            right = x[i + m:i + 2 * m]
            x[i:i + m] = (left + right) % 2
            x[i + m:i + 2 * m] = right
        m *= 2
    return x

def g2(u0, u1):
    out = [0] * (2 * len(u0))
    for i in range(len(u0)):
        out[i] = (u0[i] + u1[i]) % 2
        out[i + len(u0)] = u1[i]
    return out

def encode_rec(u):
    if len(u) == 1:
        return u
    half = len(u) // 2
    return g2(encode_rec(u[:half]), encode_rec(u[half:]))

enc_rec = np.array(encode_rec(u_seq))
print("Encoded (recursive):", enc_rec)

enc_iter = encode_iter(u_seq, n_bits)
print("Encoded (iterative):", enc_iter)
