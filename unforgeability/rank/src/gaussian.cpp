#include "../include/gaussian.h"

inline uint getValue( uint64_t *row, uint col ){
    return ( row[col/64] >> (63 - (col%64)) ) & 1 ;
}

inline uint getCol( uint col ){
    return col/64;
}

int getBoolRank( uint64_t **matrix, uint rows, uint cols) {

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    uint tot_cols = ceil( cols/ 64.0 );

    int rank = 0;

    std::cout<< "Start computing the rank for matrix of size " << rows << " x " << cols << std::endl;

    for (uint col = 0, row = 0; col < cols && row < rows; ++col) {

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        std::cout << "fractional rank progress ...   "
            << duration.count()<<"s"
            << "   " << col <<" / " << cols <<  "    " << row << " / " << rows << std::endl;

        uint pivot_row = row;
        for (uint i = row + 1; i < rows; ++i) {
            if( getValue( matrix[i], col ) ){
                pivot_row = i;
                break;
            }
        }

        if ( 0 == getValue( matrix[pivot_row], col )  ){
            std::cout<<"zero row\n";
            continue;
        }

        if (pivot_row != row) {
            auto *temp = matrix[row];
            matrix[row] = matrix[pivot_row];
            matrix[pivot_row] = temp;
        }

        #pragma omp parallel for
        for (uint i = row + 1; i < rows; ++i) {

            if( getValue( matrix[i], col ) ){
                for (uint j = getCol(col) /*0*/; j < tot_cols; ++j) {
                    matrix[i][j] ^= matrix[row][j];
                }
            }
        }

        ++row;
        ++rank;
    }
    std::cout<<"\n";

    return rank;
}
