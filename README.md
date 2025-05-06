# Polar Codes: An Introduction and Implementation

Polar codes are a groundbreaking class of error-correcting codes introduced by Erdal Arıkan in 2009. They are the first family of codes proven to achieve the theoretical limits of channel capacity for symmetric binary-input memoryless channels, making them a cornerstone of modern communication systems like 5G. This project provides an educational implementation of polar codes, focusing on their core principles, encoding/decoding algorithms, and performance analysis.

# Polar Codes: Encoding and Decoding

This project implements a polar code system, including encoding, Successive Cancellation (SC) decoding, and Successive Cancellation List (SCL) decoding. Polar codes are the first class of codes proven to achieve the Shannon capacity for a wide class of channels, making them highly efficient for modern communication systems.

# Encoding
The function polar_transform_iter carries out the encoding of the message using the Arikan polar transform.
The input message vector u of length N (a power of 2) is encoded using a structured XOR network derived from the Kronecker power of the base matrix 
𝐺2=[1 0
    1 1]
The encoding is done in-place and uses iterative bitwise operations to produce the encoded codeword x.

Time Complexity: O(N log N)
Efficient due to the recursive, divide-and-conquer style butterfly computation similar to FFT.

# SC Decoding (Successive Cancellation)
Implemented in the sc_decode function.
SC decoding processes received channel LLRs sequentially, computing intermediate LLRs via recursive f_function and g_function calls.
It makes a decision for each bit using:
The sign of the LLR if the bit is not frozen.
A fixed 0 if the bit is frozen.

Time Complexity: O(N log N)
This is due to the recursive LLR tree computation across all bit positions.

#  SCL Decoding (Successive Cancellation List)
Implemented in the scl_decode function.
SCL decoding maintains a list of L decoding paths instead of a single one.
At each unfrozen bit position, every active path is forked into two (bit = 0 and bit = 1), and path metrics are updated.
Only the L most likely paths are kept at each stage using path metric sorting.
At the end, the most likely codeword from the list is selected.

Time Complexity: O(L · N log N)
The added factor of L reflects the parallel tracking of L decoding candidates.


Compatibility with BER/Success Rate Testing: The code includes performance metrics tracking (e.g., count_successes) to evaluate decoding accuracy over many Monte Carlo trials.

