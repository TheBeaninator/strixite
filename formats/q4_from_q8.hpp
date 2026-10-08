#pragma once

// Q8 -> Q4 requantization (strixite PR #2 by @TheBeaninator):
// the MTP draft scores the LM head's first mtp_vocab rows every draft call, and those rows are Q8 in the served layout
// (178 MB at 65,536 rows). The draft only proposes tokens - the verify decides every one - so it can read a cheaper Q4
// copy (94 MB) made once at load.
//
// The copy is defined as: dequantize the Q8 rows exactly (formats/q8 dequantize_q8, the values the Q8 kernel uses),
// then quantize them as formats/q4 quantize_q4 does with group G. It is not the Q4 a converter would make from the
// BF16 checkpoint (that one starts from the original values); it is the Q4 nearest to what the engine actually serves.
//
// Threads split the rows into contiguous chunks: Q4 groups run along K inside a row, so chunks quantize independently
// and append_rows_q4 makes the result byte-identical to one call over all rows (checked in tests/test_q4_format.cpp).

#include "formats/q4.hpp"
#include "formats/q8.hpp"

#include <cstdint>

namespace strix {

// src: a Q8 weight (check_q8-consistent); G: the Q4 group size (32, 64 or 128) with src.K % G == 0; threads: 1..64
// workers over row chunks (fewer are used when there are fewer rows). Returns the Q4 weight of src.N rows x src.K.
// Throws, naming the parameter and the values, on an inconsistent src, an unsupported G, a K that G doesn't divide,
// a thread count out of range, or a failure in any worker (the first worker's message is rethrown).
Q4Weight quantize_q4_from_q8(const Q8Weight &src, int64_t G, int threads);

}  // namespace strix
