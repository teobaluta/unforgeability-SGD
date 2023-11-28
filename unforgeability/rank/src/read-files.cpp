#include "../include/read-files.h"

inline uint getCol( uint col ){
    return col/64;
}


bool readMatrix( std::string filepath, uint64_t **m, uint rows, uint cols ){

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    for( uint i=0; i< rows; i++)
        memset( m[i], 0, ceil( cols / 64.0 )  * sizeof( uint64_t) );

    char *buffer = new char[cols];

    for( uint i=0; i< rows; i++){

        file.read(buffer, cols);

        for( uint j=0; j< cols; j++)
            m[i][ getCol(j)] ^= static_cast<uint64_t>(buffer[j]-48) << (63 - ( j % 64) );

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        std::cout<<"Read row " << duration.count() << "s    " << i << " / " << rows << "\r" << std::flush;
    }
    std::cout<<std::endl;

    return true;
}

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


bool readLSBFromFile( uint bit, std::string filepath, uint64_t **m, uint rows, uint cols ){

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    for( uint i=0; i< rows; i++)
        memset( m[i], 0, ceil( cols / 64.0 ) * sizeof( uint64_t) );

    for( uint i=0; i< rows; i++){

        double f;
        for( uint j=0; j< cols; j++){
            file >> f;
            int b = getLSB( f, bit );
            m[i][ getCol(j)] ^= static_cast<uint64_t>(b) << (63 - ( j % 64) );
        }


        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        std::cout<<"Read row " << duration.count() << "s    " << i << " / " << rows << "\r" << std::flush;
    }
    std::cout<<std::endl;

    return true;

}

