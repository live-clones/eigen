#include <immintrin.h>

__m128h test_set_sh(unsigned short val) {
    return _mm_set_sh(val);
}

__m128h test_fma(__m128h a, __m128h b, __m128h c) {
    return _mm_fmadd_ph(a, b, c);
}
