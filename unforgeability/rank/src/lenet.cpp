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

    // M = 400
    //uint rows   = 25600;
    // M = 600
    uint rows   = 38400;
    // M = 800
    //uint rows = 51200;
    uint cols   = 61706;

    // get files
    std::vector<std::string> target_files;
    const std::string dir_path = argv[1];
    for (const auto &entry : fs::directory_iterator(dir_path)) {
        std::string fn = dir_path + "/" + entry.path().filename().string();
        if( fs::file_size(fn) != rows*cols ){
            std::cout<<"Incorrect size of file " << fn << " . Expected " << rows*cols <<" got " << fs::file_size(fn) << std::endl;
            continue;
        }
        std::cout<<"Candidate " << entry.path().filename().string() << std::endl;
        target_files.push_back( fn );
    }

    if( 0 ==  target_files.size() )
        return 0;

    // allocate matrix
    uint64_t **M = new uint64_t*[rows];
    uint tcols  = ceil( cols / 64.0 );
    for( uint i=0; i< rows; i++){
        M[i] = new uint64_t[tcols];
        memset( M[i], 0, tcols * sizeof( uint64_t) );
    }

    // process all files in the directory
    for( auto &f: target_files ){
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        std::cout<<"Checking " << rows << " " << tcols << "  :   " << f << std::endl;

        // read matrix from file
        if( !readMatrix( f, M, rows, cols ) ) {
            std::cout <<"Cannot read the matrix \n";
            continue;
        }

        // get rank
        uint rank = getBoolRank( M, rows, cols );
        std::cout<<"rank: " << rank << "\n";

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        std::ofstream outfile;
        outfile.open("result-lenet.txt", std::ios::app);
        outfile << rank <<" " << duration.count()<<" " << f << "\n";
        outfile.close();
    }

    for (uint i = 0; i < rows; i++)
        delete [] M[i];
    delete [] M;

    return 0;
}
