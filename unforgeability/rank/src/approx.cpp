#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>

#include "../include/gaussian.h"
#include "../include/read-files.h"


namespace fs = std::filesystem;

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>\n";
        return 1;
    }

    uint rows   = 25600;
    uint cols   = 61706;

    // allocate matrix
    uint64_t **M = new uint64_t*[rows];
    uint tcols  = ceil( cols / 64.0 );
    for( uint i=0; i< rows; i++){
        M[i] = new uint64_t[tcols];
        memset( M[i], 0, tcols * sizeof( uint64_t) );
    }


    // process all files in the directory
    const std::string dir_path = argv[1];
    for (const auto &entry : fs::directory_iterator(dir_path)) {
        std::string fn = dir_path + "/" + entry.path().filename().string();

        for( uint bit = 1; bit < 25; bit ++ ){
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

            // read from file
            if( !readLSBFromFile( bit, fn, M, rows, cols ) ){
                std::cout <<"Cannot read the matrix \n";
                continue;
            }

            // get rank
            uint rank = getBoolRank( M, rows, cols );

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
            std::ofstream outfile;
            outfile.open("result-approx.txt", std::ios::app);
            outfile << bit << " " << rank <<" " << duration.count()<<" " << fn << "\n";
            outfile.close();
        }
    }

    for (uint i = 0; i < rows; i++)
        delete [] M[i];
    delete [] M;



    return 0;
}
