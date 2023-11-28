#ifndef READ_FILES_H
#define READ_FILES_H

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstring>

bool readMatrix( std::string filepath, uint64_t **m, uint rows, uint cols );
bool readLSBFromFile( uint bit, std::string filepath, uint64_t **m, uint rows, uint cols );

#endif