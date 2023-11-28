#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>

typedef unsigned int uint;

int getLSB(double d, uint prec_dec) {
    union {
        double d;
        uint64_t i;
    } u;
    u.d = d;
    int expo = ((u.i >> 52) & 0x7ff) - 1023;
    uint64_t mant = u.i & 0xFFFFFFFFFFFFFull;
    uint64_t n = (1ull << 52) ^ mant;
    if (expo >= 0) n <<= expo;
    else n >>= -expo;
    uint64_t part_dec = (n & 0xFFFFFFFFFFFFFull) >> (52 - prec_dec);
    return part_dec & 1;
}

void free_pointer(char *str) {
    //printf("Freeeing pointer %p\n", str);
    free(str);
}

void* getGradsLSB(double* grads, uint prec_dec, uint64_t rows, uint64_t cols){
    //printf("rows: %ld cols: %ld", rows, cols);
    char* str = (char*) malloc((rows * cols + 1) * sizeof(char));
    for(uint64_t i = 0; i < rows*cols; i++){
        int lsb = getLSB(grads[i], prec_dec);
        str[i] = '0' + lsb;

    }
    str[rows*cols] = '\0';
    //printf("Pointer %p\n", str);
    return str;
}
