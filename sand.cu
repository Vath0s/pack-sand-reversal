///nvcc -o fil main.cu -O3 -m=64 -arch=compute_61 -code=sm_61 -Xptxas -allow-expensive-optimizations=true -Xptxas -v
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <inttypes.h>
#include <bitset>
#include <iostream>
#include <vector>
#include <map>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <mutex>
#include <time.h>
#include "lcg.h"

#ifdef BOINC
  #include "boinc_api.h"
#if defined _WIN32 || defined _WIN64
  #include "boinc_win.h"
#endif
#endif

uint64_t millis() {return (std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch())).count();}


#define GPU_ASSERT(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
    exit(code);
  }
}



// ===== LCG IMPLEMENTATION ===== //

namespace java_lcg { //region Java LCG
    #define Random uint64_t
    #define RANDOM_MULTIPLIER 0x5DEECE66DULL
    #define RANDOM_ADDEND 0xBULL
    #define RANDOM_MASK ((1ULL << 48u) - 1)
    #define get_random(seed) ((Random)((seed ^ RANDOM_MULTIPLIER) & RANDOM_MASK))


    __host__ __device__ __forceinline__ static int32_t random_next(Random *random, int bits) {
        *random = (*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
        return (int32_t) (*random >> (48u - bits));
    }
    __device__ __forceinline__ static int32_t random_next_int(Random *random, const uint16_t bound) {
        int32_t r = random_next(random, 31);
        const uint16_t m = bound - 1u;
        if ((bound & m) == 0) {
            r = (int32_t) ((bound * (uint64_t) r) >> 31u);
        } else {
            for (int32_t u = r;
                 u - (r = u % bound) + m < 0;
                 u = random_next(random, 31));
        }
        return r;
    }
    
    __device__ __host__ __forceinline__ static int32_t random_next_int_nonpow(Random *random, const uint16_t bound) {
        int32_t r = random_next(random, 31);
        const uint16_t m = bound - 1u;
        for (int32_t u = r;
             u - (r = u % bound) + m < 0;
             u = random_next(random, 31));
      return r;
    }
    __host__ __device__ __forceinline__ static double next_double(Random *random) {
        return (double) ((((uint64_t) ((uint32_t) random_next(random, 26)) << 27u)) + random_next(random, 27)) / (double)(1ULL << 53);
    }
    __host__ __device__ __forceinline__ static uint64_t random_next_long (Random *random) {
        return (((uint64_t)random_next(random, 32)) << 32u) + (int32_t)random_next(random, 32);
    }
    __host__ __device__ __forceinline__ static void advance2(Random *random) {
        *random = (*random * 0xBB20B4600A69LLU + 0x40942DE6BALLU) & RANDOM_MASK;
    }
    __host__ __device__ __forceinline__ static void advance3759(Random *random) {
        *random = (*random * 0x6FE85C031F25LLU + 0x8F50ECFF899LLU) & RANDOM_MASK;
    }

}
using namespace java_lcg;


namespace device_intrinsics { //region DEVICE INTRINSICS
    #define DEVICE_STATIC_INTRINSIC_QUALIFIERS  static __device__ __forceinline__

    #if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
    #define PXL_GLOBAL_PTR   "l"
    #else
    #define PXL_GLOBAL_PTR   "r"
    #endif

    DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_local_l1(const void* const ptr)
    {
      asm("prefetch.local.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
    }

    DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_uniform(const void* const ptr)
    {
      asm("prefetchu.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
    }

    DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_local_l2(const void* const ptr)
    {
      asm("prefetch.local.L2 [%0];" : : PXL_GLOBAL_PTR(ptr));
    }

    #if __CUDA__ < 10
    #define __ldg(ptr) (*(ptr))
    #endif
}
using namespace device_intrinsics;






#define BLOCK_SIZE (128)
//#define BLOCK_SIZE (128)
#define WORK_SIZE_BITS 16
#define SEEDS_PER_CALL ((1ULL << (WORK_SIZE_BITS)) * (BLOCK_SIZE))
//#define SEEDS_PER_CALL 8000000




//Specifying where the (1 = dirt/grass, 0 = sand) is

// This will match the seed 76261196830436 (not pack.png ofc)
// Double match: 76261206560653 (almost 100% confirmed, sans very last bit of sand in first match)
// Triple match: 76273693341674 (100% match)
#define CHUNK_X 6
#define CHUNK_Z -1

#define INNER_X_START 4
#define INNER_Z_START 0

#define INNER_X_END 13
#define INNER_Z_END 2
__constant__ uint8_t DIRT_HEIGHT_2D[INNER_Z_END - INNER_Z_START + 1][INNER_X_END - INNER_X_START + 1] = {{1,15,15,15,1,15,0,15,15,15},
                                                                                                         {15,1,15,15,15,1,15,1,15,15},
                                                                                                         {15,15,1,1,15,15,1,1,1,0}};
__constant__ double LocalNoise2D[INNER_Z_END - INNER_Z_START + 1][INNER_X_END - INNER_X_START + 1];

#define EARLY_RETURN (INNER_Z_END * 16 + INNER_X_END)


#define CHUNK_X_2 6
#define CHUNK_Z_2 -2

#define INNER_X_START_2 0
#define INNER_Z_START_2 6

#define INNER_X_END_2 9
#define INNER_Z_END_2 15

__constant__ uint8_t DIRT_HEIGHT_2D_2[INNER_Z_END_2 - INNER_Z_START_2 + 1][INNER_X_END_2 - INNER_X_START_2 + 1] = {{0,15,15,15,15,15,15,15,15,15},
                                                                                                                   {15,0,0,15,15,15,15,15,15,15},
                                                                                                                   {0,15,15,0,15,15,15,15,15,15},
                                                                                                                   {15,1,15,15,0,15,15,15,15,15},
                                                                                                                   {15,15,0,15,15,0,15,15,15,15},
                                                                                                                   {15,15,15,0,15,15,0,15,15,15},
                                                                                                                   {0,15,15,15,15,0,0,15,15,15},
                                                                                                                   {0,0,15,15,15,15,0,0,0,15},
                                                                                                                   {15,15,0,0,15,15,15,0,15,0}};
__constant__ double LocalNoise2D_2[INNER_Z_END_2 - INNER_Z_START_2 + 1][INNER_X_END_2 - INNER_X_START_2 + 1];


#define CHUNK_X_3 5
#define CHUNK_Z_3 -1

#define INNER_X_START_3 4
#define INNER_Z_START_3 0

#define INNER_X_END_3 15
#define INNER_Z_END_3 10

__constant__ uint8_t DIRT_HEIGHT_2D_3[INNER_Z_END_3 - INNER_Z_START_3 + 1][INNER_X_END_3 - INNER_X_START_3 + 1] = {{1,1,15,15,15,15,15,15,15,15,0,15},
                                                                                                                   {15,15,15,15,15,15,15,15,15,15,0,15},
                                                                                                                   {15,15,15,15,15,15,15,15,15,15,15,0},
                                                                                                                   {15,15,15,0,15,15,15,15,15,15,15,0},
                                                                                                                   {15,15,15,1,15,15,15,15,15,15,15,15},
                                                                                                                   {15,15,15,0,15,15,15,15,15,15,15,0},
                                                                                                                   {15,15,15,15,15,15,15,15,15,15,15,15},
                                                                                                                   {15,15,0,15,15,15,15,15,15,15,15,15},
                                                                                                                   {15,15,1,15,15,15,15,15,15,15,15,15},
                                                                                                                   {15,15,15,1,15,15,15,15,15,15,15,15},
                                                                                                                   {15,15,15,0,15,15,15,15,15,15,15,15}};
__constant__ double LocalNoise2D_3[INNER_Z_END_3 - INNER_Z_START_3 + 1][INNER_X_END_3 - INNER_X_START_3 + 1];
/*
//Old test: matches 104703450999364
#define CHUNK_X 2
#define CHUNK_Z 11

#define INNER_X_START 2
#define INNER_Z_START 0

#define INNER_X_END 11
#define INNER_Z_END 0


__constant__ uint8_t DIRT_HEIGHT_2D[INNER_Z_END - INNER_Z_START + 1][INNER_X_END - INNER_X_START + 1] = {{0,15,0,1,0,15,15,15,15,1}};
__constant__ double LocalNoise2D[INNER_Z_END - INNER_Z_START + 1][INNER_X_END - INNER_X_START + 1];
*/



//The generation of the simplex layers and noise
namespace noise { //region Simplex layer gen
    /* End of constant for simplex noise*/
    
    struct Octave {
        double xo;
        double yo;
        double zo;
        uint8_t permutations[256];
    };

    __shared__ uint8_t permutations[256][BLOCK_SIZE];


    #define getValue(array, index) array[index][threadIdx.x]
    #define setValue(array, index, value) array[index][threadIdx.x] = value


    __device__ static inline void setupNoise(const uint8_t nbOctaves, Random *random, Octave resultArray[]) {
        for (int j = 0; j < nbOctaves; ++j) {
            __prefetch_local_l2(&resultArray[j]);
            resultArray[j].xo = next_double(random) * 256.0;
            resultArray[j].yo = next_double(random) * 256.0;
            resultArray[j].zo = next_double(random) * 256.0;
            
            #pragma unroll
            for(int w = 0; w<256; w++) {
                setValue(permutations, w, w);
            }
            for(int index = 0; index<256; index++) {
                uint32_t randomIndex = random_next_int(random, 256ull - index) + index;
                //if (randomIndex != index) {
                    // swap
                    uint8_t v1 = getValue(permutations,index);
                    //uint8_t v2 = getValue(permutations,randomIndex);
                    setValue(permutations,index, getValue(permutations,randomIndex));
                    setValue(permutations, randomIndex, v1);
                //}
            }
            #pragma unroll
            for(int c = 0; c<256;c++) {
                __prefetch_local_l1(&(resultArray[j].permutations[c+1]));
                resultArray[j].permutations[c] = getValue(permutations,c);
            }
            //resultArray[j].xo = xo;
            //resultArray[j].yo = yo;
            //resultArray[j].zo = zo;
        }
    }
    __device__ static inline void SkipNoiseGen(const uint8_t nbOctaves, Random* random) {
        for (int j = 0; j < nbOctaves; ++j) {
            lcg::advance<2*3>(*random);
            for(int index = 0; index<256; index++) {
                random_next_int(random, 256ull - index);
            }
        }
    }
    
    __device__ static inline double lerp(double x, double a, double b) {
        return a + x * (b - a);
    }

    __device__ static inline double grad(uint8_t hash, double x, double y, double z) {
        switch (hash & 0xFu) {
            case 0x0:
                return x + y;
            case 0x1:
                return -x + y;
            case 0x2:
                return x - y;
            case 0x3:
                return -x - y;
            case 0x4:
                return x + z;
            case 0x5:
                return -x + z;
            case 0x6:
                return x - z;
            case 0x7:
                return -x - z;
            case 0x8:
                return y + z;
            case 0x9:
                return -y + z;
            case 0xA:
                return y - z;
            case 0xB:
                return -y - z;
            case 0xC:
                return y + x;
            case 0xD:
                return -y + z;
            case 0xE:
                return y - x;
            case 0xF:
                return -y - z;
            default:
                return 0; // never happens
        }
    }


    __device__ static inline void generateNormalPermutations(double *buffer, double x, double y, double z, int sizeX, int sizeY, int sizeZ, double noiseFactorX, double noiseFactorY, double noiseFactorZ, double octaveSize, Random* random) {
        double xo = lcg::next_double(*random) * 256.0;
		double yo = lcg::next_double(*random) * 256.0;
		double zo = lcg::next_double(*random) * 256.0;
		//Setup the permutation fresh xD
		#pragma unroll
		for(int w = 0; w<256; w++) {
			setValue(permutations, w, w);
		}
		for(int index = 0; index<256; index++) {
			uint32_t randomIndex = lcg::dynamic_next_int(*random, 256ull - index) + index;
			//if (randomIndex != index) {
				// swap
				uint8_t v1 = getValue(permutations,index);
				uint8_t v2 = getValue(permutations,randomIndex);
				setValue(permutations,index, v2);
				setValue(permutations, randomIndex, v1);
			//}
		}
		
		double octaveWidth = 1.0 / octaveSize;
        int32_t i2 = -1;
        double x1 = 0.0;
        double x2 = 0.0;
        double xx1 = 0.0;
        double xx2 = 0.0;
        double t;
        double w;
        int columnIndex = 0;
        for (int X = 0; X < sizeX; X++) {
            double xCoord = (x + (double) X) * noiseFactorX + xo;
            auto clampedXcoord = (int32_t) xCoord;
            if (xCoord < (double) clampedXcoord) {
                clampedXcoord--;
            }
            auto xBottoms = (uint8_t) ((uint32_t) clampedXcoord & 0xffu);
            xCoord -= clampedXcoord;
            t = xCoord * 6 - 15;
            w = (xCoord * t + 10);
            double fadeX = xCoord * xCoord * xCoord * w;
            for (int Z = 0; Z < sizeZ; Z++) {
                double zCoord = zo;
                auto clampedZCoord = (int32_t) zCoord;
                if (zCoord < (double) clampedZCoord) {
                    clampedZCoord--;
                }
                auto zBottoms = (uint8_t) ((uint32_t) clampedZCoord & 0xffu);
                zCoord -= clampedZCoord;
                t = zCoord * 6 - 15;
                w = (zCoord * t + 10);
                double fadeZ = zCoord * zCoord * zCoord * w;
                for (int Y = 0; Y < sizeY; Y++) {
                    double yCoords = (y + (double) Y) * noiseFactorY + yo;
                    auto clampedYCoords = (int32_t) yCoords;
                    if (yCoords < (double) clampedYCoords) {
                        clampedYCoords--;
                    }
                    auto yBottoms = (uint8_t) ((uint32_t) clampedYCoords & 0xffu);
                    yCoords -= clampedYCoords;
                    t = yCoords * 6 - 15;
                    w = yCoords * t + 10;
                    double fadeY = yCoords * yCoords * yCoords * w;
                    // ZCoord

                    if (Y == 0 || yBottoms != i2) { // this is wrong on so many levels, same ybottoms doesnt mean x and z were the same...
						i2 = yBottoms;
						uint16_t k2 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)(xBottoms& 0xffu)) + yBottoms)& 0xffu)) + zBottoms;
						uint16_t l2 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)(xBottoms& 0xffu)) + yBottoms + 1u )& 0xffu)) + zBottoms;
						uint16_t k3 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)((xBottoms + 1u)& 0xffu)) + yBottoms )& 0xffu)) + zBottoms;
						uint16_t l3 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)((xBottoms + 1u)& 0xffu)) + yBottoms + 1u) & 0xffu)) + zBottoms;
						x1 = lerp(fadeX, grad(getValue(permutations,(uint8_t)(k2& 0xffu)), xCoord, yCoords, zCoord), grad(getValue(permutations,(uint8_t)(k3& 0xffu)), xCoord - 1.0, yCoords, zCoord));
						x2 = lerp(fadeX, grad(getValue(permutations,(uint8_t)(l2& 0xffu)), xCoord, yCoords - 1.0, zCoord), grad(getValue(permutations,(uint8_t)(l3& 0xffu)), xCoord - 1.0, yCoords - 1.0, zCoord));
						xx1 = lerp(fadeX, grad(getValue(permutations,(uint8_t)((k2+1u)& 0xffu)), xCoord, yCoords, zCoord - 1.0), grad(getValue(permutations,(uint8_t)((k3+1u)& 0xffu)), xCoord - 1.0, yCoords, zCoord - 1.0));
						xx2 = lerp(fadeX, grad(getValue(permutations,(uint8_t)((l2+1u)& 0xffu)), xCoord, yCoords - 1.0, zCoord - 1.0), grad(getValue(permutations,(uint8_t)((l3+1u)& 0xffu)), xCoord - 1.0, yCoords - 1.0, zCoord - 1.0));
					}

                    if (columnIndex%16 >= INNER_X_START && columnIndex%16 <= INNER_X_END &&
                        DIRT_HEIGHT_2D[columnIndex/16 - INNER_Z_START][columnIndex%16 - INNER_X_START] != 15){
                        double y1 = lerp(fadeY, x1, x2);
                        double y2 = lerp(fadeY, xx1, xx2);
                        (buffer)[columnIndex] = (buffer)[columnIndex] + lerp(fadeZ, y1, y2) * octaveWidth;
                    }

                    if (columnIndex == EARLY_RETURN) return;
                    
                    columnIndex++;

                }
            }
        }
    }


    __device__ static inline void generateNormalPermutations_2(double *buffer, double x, double y, double z, int sizeX, int sizeY, int sizeZ, double noiseFactorX, double noiseFactorY, double noiseFactorZ, double octaveSize, Random* random) {
        double xo = lcg::next_double(*random) * 256.0;
		double yo = lcg::next_double(*random) * 256.0;
		double zo = lcg::next_double(*random) * 256.0;
		//Setup the permutation fresh xD
		#pragma unroll
		for(int w = 0; w<256; w++) {
			setValue(permutations, w, w);
		}
		for(int index = 0; index<256; index++) {
			uint32_t randomIndex = lcg::dynamic_next_int(*random, 256ull - index) + index;
			//if (randomIndex != index) {
				// swap
				uint8_t v1 = getValue(permutations,index);
				uint8_t v2 = getValue(permutations,randomIndex);
				setValue(permutations,index, v2);
				setValue(permutations, randomIndex, v1);
			//}
		}
		double octaveWidth = 1.0 / octaveSize;
        int32_t i2 = -1;
        double x1 = 0.0;
        double x2 = 0.0;
        double xx1 = 0.0;
        double xx2 = 0.0;
        double t;
        double w;
        int columnIndex = 0;
        for (int X = 0; X < sizeX; X++) {
            double xCoord = (x + (double) X) * noiseFactorX + xo;
            auto clampedXcoord = (int32_t) xCoord;
            if (xCoord < (double) clampedXcoord) {
                clampedXcoord--;
            }
            auto xBottoms = (uint8_t) ((uint32_t) clampedXcoord & 0xffu);
            xCoord -= clampedXcoord;
            t = xCoord * 6 - 15;
            w = (xCoord * t + 10);
            double fadeX = xCoord * xCoord * xCoord * w;
            for (int Z = 0; Z < sizeZ; Z++) {
                double zCoord = zo;
                auto clampedZCoord = (int32_t) zCoord;
                if (zCoord < (double) clampedZCoord) {
                    clampedZCoord--;
                }
                auto zBottoms = (uint8_t) ((uint32_t) clampedZCoord & 0xffu);
                zCoord -= clampedZCoord;
                t = zCoord * 6 - 15;
                w = (zCoord * t + 10);
                double fadeZ = zCoord * zCoord * zCoord * w;
                for (int Y = 0; Y < sizeY; Y++) {
                    double yCoords = (y + (double) Y) * noiseFactorY + yo;
                    auto clampedYCoords = (int32_t) yCoords;
                    if (yCoords < (double) clampedYCoords) {
                        clampedYCoords--;
                    }
                    auto yBottoms = (uint8_t) ((uint32_t) clampedYCoords & 0xffu);
                    yCoords -= clampedYCoords;
                    t = yCoords * 6 - 15;
                    w = yCoords * t + 10;
                    double fadeY = yCoords * yCoords * yCoords * w;
                    // ZCoord

                    if (Y == 0 || yBottoms != i2) { // this is wrong on so many levels, same ybottoms doesnt mean x and z were the same...
						i2 = yBottoms;
						uint16_t k2 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)(xBottoms& 0xffu)) + yBottoms)& 0xffu)) + zBottoms;
						uint16_t l2 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)(xBottoms& 0xffu)) + yBottoms + 1u )& 0xffu)) + zBottoms;
						uint16_t k3 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)((xBottoms + 1u)& 0xffu)) + yBottoms )& 0xffu)) + zBottoms;
						uint16_t l3 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)((xBottoms + 1u)& 0xffu)) + yBottoms + 1u) & 0xffu)) + zBottoms;
						x1 = lerp(fadeX, grad(getValue(permutations,(uint8_t)(k2& 0xffu)), xCoord, yCoords, zCoord), grad(getValue(permutations,(uint8_t)(k3& 0xffu)), xCoord - 1.0, yCoords, zCoord));
						x2 = lerp(fadeX, grad(getValue(permutations,(uint8_t)(l2& 0xffu)), xCoord, yCoords - 1.0, zCoord), grad(getValue(permutations,(uint8_t)(l3& 0xffu)), xCoord - 1.0, yCoords - 1.0, zCoord));
						xx1 = lerp(fadeX, grad(getValue(permutations,(uint8_t)((k2+1u)& 0xffu)), xCoord, yCoords, zCoord - 1.0), grad(getValue(permutations,(uint8_t)((k3+1u)& 0xffu)), xCoord - 1.0, yCoords, zCoord - 1.0));
						xx2 = lerp(fadeX, grad(getValue(permutations,(uint8_t)((l2+1u)& 0xffu)), xCoord, yCoords - 1.0, zCoord - 1.0), grad(getValue(permutations,(uint8_t)((l3+1u)& 0xffu)), xCoord - 1.0, yCoords - 1.0, zCoord - 1.0));
					}

                    if (columnIndex%16 >= INNER_X_START_2 && columnIndex%16 <= INNER_X_END_2 &&
                        DIRT_HEIGHT_2D_2[columnIndex/16 - INNER_Z_START_2][columnIndex%16 - INNER_X_START_2] != 15){
                        double y1 = lerp(fadeY, x1, x2);
                        double y2 = lerp(fadeY, xx1, xx2);
                        (buffer)[columnIndex] = (buffer)[columnIndex] + lerp(fadeZ, y1, y2) * octaveWidth;
                    }
                    
                    columnIndex++;

                }
            }
        }
    }

    __device__ static inline void generateNormalPermutations_3(double *buffer, double x, double y, double z, int sizeX, int sizeY, int sizeZ, double noiseFactorX, double noiseFactorY, double noiseFactorZ, double octaveSize, Random* random) {
        double xo = lcg::next_double(*random) * 256.0;
		double yo = lcg::next_double(*random) * 256.0;
		double zo = lcg::next_double(*random) * 256.0;
		//Setup the permutation fresh xD
		#pragma unroll
		for(int w = 0; w<256; w++) {
			setValue(permutations, w, w);
		}
		for(int index = 0; index<256; index++) {
			uint32_t randomIndex = lcg::dynamic_next_int(*random, 256ull - index) + index;
			//if (randomIndex != index) {
				// swap
				uint8_t v1 = getValue(permutations,index);
				uint8_t v2 = getValue(permutations,randomIndex);
				setValue(permutations,index, v2);
				setValue(permutations, randomIndex, v1);
			//}
		}
		double octaveWidth = 1.0 / octaveSize;
        int32_t i2 = -1;
        double x1 = 0.0;
        double x2 = 0.0;
        double xx1 = 0.0;
        double xx2 = 0.0;
        double t;
        double w;
        int columnIndex = 0;
        for (int X = 0; X < sizeX; X++) {
            double xCoord = (x + (double) X) * noiseFactorX + xo;
            auto clampedXcoord = (int32_t) xCoord;
            if (xCoord < (double) clampedXcoord) {
                clampedXcoord--;
            }
            auto xBottoms = (uint8_t) ((uint32_t) clampedXcoord & 0xffu);
            xCoord -= clampedXcoord;
            t = xCoord * 6 - 15;
            w = (xCoord * t + 10);
            double fadeX = xCoord * xCoord * xCoord * w;
            for (int Z = 0; Z < sizeZ; Z++) {
                double zCoord = zo;
                auto clampedZCoord = (int32_t) zCoord;
                if (zCoord < (double) clampedZCoord) {
                    clampedZCoord--;
                }
                auto zBottoms = (uint8_t) ((uint32_t) clampedZCoord & 0xffu);
                zCoord -= clampedZCoord;
                t = zCoord * 6 - 15;
                w = (zCoord * t + 10);
                double fadeZ = zCoord * zCoord * zCoord * w;
                for (int Y = 0; Y < sizeY; Y++) {
                    double yCoords = (y + (double) Y) * noiseFactorY + yo;
                    auto clampedYCoords = (int32_t) yCoords;
                    if (yCoords < (double) clampedYCoords) {
                        clampedYCoords--;
                    }
                    auto yBottoms = (uint8_t) ((uint32_t) clampedYCoords & 0xffu);
                    yCoords -= clampedYCoords;
                    t = yCoords * 6 - 15;
                    w = yCoords * t + 10;
                    double fadeY = yCoords * yCoords * yCoords * w;
                    // ZCoord

                    if (Y == 0 || yBottoms != i2) { // this is wrong on so many levels, same ybottoms doesnt mean x and z were the same...
						i2 = yBottoms;
						uint16_t k2 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)(xBottoms& 0xffu)) + yBottoms)& 0xffu)) + zBottoms;
						uint16_t l2 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)(xBottoms& 0xffu)) + yBottoms + 1u )& 0xffu)) + zBottoms;
						uint16_t k3 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)((xBottoms + 1u)& 0xffu)) + yBottoms )& 0xffu)) + zBottoms;
						uint16_t l3 = getValue(permutations,(uint8_t)((uint16_t)(getValue(permutations,(uint8_t)((xBottoms + 1u)& 0xffu)) + yBottoms + 1u) & 0xffu)) + zBottoms;
						x1 = lerp(fadeX, grad(getValue(permutations,(uint8_t)(k2& 0xffu)), xCoord, yCoords, zCoord), grad(getValue(permutations,(uint8_t)(k3& 0xffu)), xCoord - 1.0, yCoords, zCoord));
						x2 = lerp(fadeX, grad(getValue(permutations,(uint8_t)(l2& 0xffu)), xCoord, yCoords - 1.0, zCoord), grad(getValue(permutations,(uint8_t)(l3& 0xffu)), xCoord - 1.0, yCoords - 1.0, zCoord));
						xx1 = lerp(fadeX, grad(getValue(permutations,(uint8_t)((k2+1u)& 0xffu)), xCoord, yCoords, zCoord - 1.0), grad(getValue(permutations,(uint8_t)((k3+1u)& 0xffu)), xCoord - 1.0, yCoords, zCoord - 1.0));
						xx2 = lerp(fadeX, grad(getValue(permutations,(uint8_t)((l2+1u)& 0xffu)), xCoord, yCoords - 1.0, zCoord - 1.0), grad(getValue(permutations,(uint8_t)((l3+1u)& 0xffu)), xCoord - 1.0, yCoords - 1.0, zCoord - 1.0));
					}

                    if (columnIndex%16 >= INNER_X_START_3 && columnIndex%16 <= INNER_X_END_3 &&
                        DIRT_HEIGHT_2D_3[columnIndex/16 - INNER_Z_START_3][columnIndex%16 - INNER_X_START_3] != 15){
                        double y1 = lerp(fadeY, x1, x2);
                        double y2 = lerp(fadeY, xx1, xx2);
                        (buffer)[columnIndex] = (buffer)[columnIndex] + lerp(fadeZ, y1, y2) * octaveWidth;
                    }
                    
                    columnIndex++;

                }
            }
        }
    }


    __device__ static inline void generateNoise(double *buffer, double chunkX, double chunkY, double chunkZ, int sizeX, int sizeY, int sizeZ, double offsetX, double offsetY, double offsetZ, Random random, int nbOctaves) {
        //memset(buffer, 0, sizeof(double) * sizeX * sizeZ * sizeY);
        double octavesFactor = 1.0;
        for (int octave = 0; octave < nbOctaves; octave++) {
            generateNormalPermutations(buffer, chunkX, chunkY, chunkZ, sizeX, sizeY, sizeZ, offsetX * octavesFactor, offsetY * octavesFactor, offsetZ * octavesFactor, octavesFactor, &random);
            octavesFactor /= 2.0;
        }
    }

    __device__ static inline void generateNoise_2(double *buffer, double chunkX, double chunkY, double chunkZ, int sizeX, int sizeY, int sizeZ, double offsetX, double offsetY, double offsetZ, Random random, int nbOctaves) {
        //memset(buffer, 0, sizeof(double) * sizeX * sizeZ * sizeY);
        double octavesFactor = 1.0;
        for (int octave = 0; octave < nbOctaves; octave++) {
            generateNormalPermutations_2(buffer, chunkX, chunkY, chunkZ, sizeX, sizeY, sizeZ, offsetX * octavesFactor, offsetY * octavesFactor, offsetZ * octavesFactor, octavesFactor, &random);
            octavesFactor /= 2.0;
        }
    }

    __device__ static inline void generateNoise_3(double *buffer, double chunkX, double chunkY, double chunkZ, int sizeX, int sizeY, int sizeZ, double offsetX, double offsetY, double offsetZ, Random random, int nbOctaves) {
        //memset(buffer, 0, sizeof(double) * sizeX * sizeZ * sizeY);
        double octavesFactor = 1.0;
        for (int octave = 0; octave < nbOctaves; octave++) {
            generateNormalPermutations_3(buffer, chunkX, chunkY, chunkZ, sizeX, sizeY, sizeZ, offsetX * octavesFactor, offsetY * octavesFactor, offsetZ * octavesFactor, octavesFactor, &random);
            octavesFactor /= 2.0;
        }
    }
}
using namespace noise;


__device__ static inline bool match(uint64_t seed) {
    seed = get_random(seed);
    //SkipNoiseGen(16+16+8, &seed);
    lcg::advance<10480>(seed);//VERY VERY DODGY
    
    
    double heightField[EARLY_RETURN+1];
    #pragma unroll
    for(uint16_t i = 0; i<EARLY_RETURN+1;i++)
        heightField[i] = 0;
    
    const double noiseFactor = 0.03125;
    generateNoise(heightField, (double) (CHUNK_X <<4), (double) (CHUNK_Z<<4), 0.0, 16, 16, 1, noiseFactor, noiseFactor, 1.0, seed, 4);

    for(uint8_t z = 0; z < INNER_Z_END - INNER_Z_START + 1; z++) {
        for(uint8_t x = 0; x < INNER_X_END - INNER_X_START + 1; x++) {
            if (DIRT_HEIGHT_2D[z][x] != 15) {
                uint8_t dirty = heightField[INNER_X_START + x + (INNER_Z_START + z) * 16] + LocalNoise2D[z][x] * 0.2 > 0.0 ? 0 : 1;
                if (dirty!=(int8_t)DIRT_HEIGHT_2D[z][x]) 
                    return false;
            }
        }
    }
    return true;
}

__device__ static inline bool match2(uint64_t seed) {
    seed = get_random(seed);
    //SkipNoiseGen(16+16+8, &seed);
    lcg::advance<10480>(seed);//VERY VERY DODGY
    
    double heightField[256];
    #pragma unroll
    for(uint16_t i = 0; i<256;i++)
        heightField[i] = 0;
    
    const double noiseFactor = 0.03125;
    generateNoise_2(heightField, (double) (CHUNK_X_2 <<4), (double) (CHUNK_Z_2<<4), 0.0, 16, 16, 1, noiseFactor, noiseFactor, 1.0, seed, 4);

    for(uint8_t z = 0; z < INNER_Z_END_2 - INNER_Z_START_2 + 1; z++) {
        for(uint8_t x = 0; x < INNER_X_END_2 - INNER_X_START_2 + 1; x++) {
            if (DIRT_HEIGHT_2D_2[z][x] != 15) {
                uint8_t dirty = heightField[INNER_X_START_2 + x + (INNER_Z_START_2 + z) * 16] + LocalNoise2D_2[z][x] * 0.2 > 0.0 ? 0 : 1;
                if (dirty!=(int8_t)DIRT_HEIGHT_2D_2[z][x]) 
                    return false;
            }
        }
    }
    return true;
}

__device__ static inline bool match3(uint64_t seed) {
    seed = get_random(seed);
    //SkipNoiseGen(16+16+8, &seed);
    lcg::advance<10480>(seed);//VERY VERY DODGY
    
    
    double heightField[256];
    #pragma unroll
    for(uint16_t i = 0; i<256;i++)
        heightField[i] = 0;
    
    const double noiseFactor = 0.03125;
    generateNoise_3(heightField, (double) (CHUNK_X_3 <<4), (double) (CHUNK_Z_3<<4), 0.0, 16, 16, 1, noiseFactor, noiseFactor, 1.0, seed, 4);

    for(uint8_t z = 0; z < INNER_Z_END_3 - INNER_Z_START_3 + 1; z++) {
        for(uint8_t x = 0; x < INNER_X_END_3 - INNER_X_START_3 + 1; x++) {
            if (DIRT_HEIGHT_2D_3[z][x] != 15) {
                uint8_t dirty = heightField[INNER_X_START_3 + x + (INNER_Z_START_3 + z) * 16] + LocalNoise2D_3[z][x] * 0.2 > 0.0 ? 0 : 1;
                if (dirty!=(int8_t)DIRT_HEIGHT_2D_3[z][x]) 
                    return false;
            }
        }
    }
    return true;
}


__global__ __launch_bounds__(BLOCK_SIZE,2) static void tempCheck(uint64_t offset, uint64_t* buffer, uint32_t* counter) {
    uint64_t seed = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (match(seed)) {
        buffer[atomicAdd(counter,1)] = seed;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE,2) static void tempCheck2(uint32_t count, uint64_t* buffer) {
    uint64_t seedIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (seedIndex>=count)
        return;
    if (!match2(buffer[seedIndex])) {
        buffer[seedIndex] = 0;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE,2) static void tempCheck3(uint32_t count, uint64_t* buffer) {
    uint64_t seedIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (seedIndex>=count)
        return;
    if (!match3(buffer[seedIndex])) {
        buffer[seedIndex] = 0;
    }
}






std::ifstream inSeeds;
std::ofstream outSeeds;

uint64_t* buffer;
uint32_t* counter;

double getNextDoubleForLocNoise(int x, int z);
void setup(int gpu_device) {
    cudaSetDevice(gpu_device);
    GPU_ASSERT(cudaPeekAtLastError());
    GPU_ASSERT(cudaDeviceSynchronize());
    
    double locNoise2D[INNER_Z_END - INNER_Z_START + 1][INNER_X_END - INNER_X_START + 1];
    for(uint8_t z = 0; z < INNER_Z_END - INNER_Z_START + 1; z++) {
        for (uint8_t x = 0; x < INNER_X_END - INNER_X_START + 1; x++) {
            locNoise2D[z][x] = getNextDoubleForLocNoise((CHUNK_X<<4) + INNER_X_START + x, (CHUNK_Z<<4) + INNER_Z_START + z);
        }
    }

    GPU_ASSERT(cudaMemcpyToSymbol(LocalNoise2D, &locNoise2D, sizeof(locNoise2D)));
    GPU_ASSERT(cudaPeekAtLastError());
    
    double locNoise2D_2[INNER_Z_END_2 - INNER_Z_START_2 + 1][INNER_X_END_2 - INNER_X_START_2 + 1];
    for(uint8_t z = 0; z < INNER_Z_END_2 - INNER_Z_START_2 + 1; z++) {
        for (uint8_t x = 0; x < INNER_X_END_2 - INNER_X_START_2 + 1; x++) {
            locNoise2D_2[z][x] = getNextDoubleForLocNoise((CHUNK_X_2<<4) + INNER_X_START_2 + x, (CHUNK_Z_2<<4) + INNER_Z_START_2 + z);
        }
    }

    GPU_ASSERT(cudaMemcpyToSymbol(LocalNoise2D_2, &locNoise2D_2, sizeof(locNoise2D_2)));
    GPU_ASSERT(cudaPeekAtLastError());
    
    double locNoise2D_3[INNER_Z_END_3 - INNER_Z_START_3 + 1][INNER_X_END_3 - INNER_X_START_3 + 1];
    for(uint8_t z = 0; z < INNER_Z_END_3 - INNER_Z_START_3 + 1; z++) {
        for (uint8_t x = 0; x < INNER_X_END_3 - INNER_X_START_3 + 1; x++) {
            locNoise2D_3[z][x] = getNextDoubleForLocNoise((CHUNK_X_3<<4) + INNER_X_START_3 + x, (CHUNK_Z_3<<4) + INNER_Z_START_3 + z);
        }
    }

    GPU_ASSERT(cudaMemcpyToSymbol(LocalNoise2D_3, &locNoise2D_3, sizeof(locNoise2D_3)));
    GPU_ASSERT(cudaPeekAtLastError());
}


time_t elapsed_chkpoint = 0;
struct checkpoint_vars {
    unsigned long long offset;
    time_t elapsed_chkpoint;
    unsigned long long vector1Size;
    unsigned long long vector2Size;
    };

int main(int argc, char *argv[]) {
    std::vector<uint64_t> filtered_once;
    std::vector<uint64_t> filtered_twice;
    int gpu_device = 0;
    uint64_t START;
    uint64_t offsetStart = 0;
    uint64_t COUNT;
	#ifdef BOINC
    BOINC_OPTIONS options;
    boinc_options_defaults(options);
	options.normal_thread_priority = true;
    boinc_init_options(&options);
    #endif
	for (int i = 1; i < argc; i += 2) {
		const char *param = argv[i];
		if (strcmp(param, "-d") == 0 || strcmp(param, "--device") == 0) {
			gpu_device = atoi(argv[i + 1]);
		} else if (strcmp(param, "-s") == 0 || strcmp(param, "--start") == 0) {
			sscanf(argv[i + 1], "%llu", &START);
		} else if (strcmp(param, "-e") == 0 || strcmp(param, "--count") == 0) {
			sscanf(argv[i + 1], "%llu", &COUNT);
		} else {
			fprintf(stderr,"Unknown parameter: %s\n", param);
		}
    }
    FILE *checkpoint_data = boinc_fopen("packpoint.txt", "rb");
    std::ifstream firstvector_data("vector1.txt");
    std::ifstream secondvector_data("vector2.txt");

    if(!checkpoint_data || !firstvector_data || !secondvector_data){
        fprintf(stderr, "No checkpoint to load\n");

    }
    else{
        #ifdef BOINC
            boinc_begin_critical_section();
        #endif

        struct checkpoint_vars data_store;
        fread(&data_store, sizeof(data_store), 1, checkpoint_data);
        offsetStart = data_store.offset;
        elapsed_chkpoint = data_store.elapsed_chkpoint;
        fprintf(stderr, "Checkpoint loaded, task time %d s, seed pos: %llu\n", elapsed_chkpoint, START);
        fclose(checkpoint_data);
        std::string str;
        while(std::getline(firstvector_data, str)){
            if(str.size() > 0){
                std::istringstream iss(str);
                uint64_t num1;
                iss >> num1;
                filtered_once.push_back(num1);
            }
        }
        while(std::getline(secondvector_data, str)){
            if(str.size() > 0){
                std::istringstream iss(str);
                uint64_t num1;
                iss >> num1;
                filtered_twice.push_back(num1);
            }
        }
        firstvector_data.close();
        secondvector_data.close();
        std::cout << filtered_once.size() << std::endl;
        std::cout << filtered_twice.size() << std::endl;
        #ifdef BOINC
            boinc_end_critical_section();
        #endif
    }
	#ifdef BOINC
	APP_INIT_DATA aid;
	boinc_get_init_data(aid);
	
	if (aid.gpu_device_num >= 0) {
		gpu_device = aid.gpu_device_num;
		fprintf(stderr,"boinc gpu %i gpuindex: %i \n", aid.gpu_device_num, gpu_device);
		} else {
		fprintf(stderr,"stndalone gpuindex %i \n", gpu_device);
	}
	#endif
    setup(gpu_device);
    uint64_t seedCount = COUNT;
    std::cout << "Processing " << seedCount << " seeds" << std::endl;

    outSeeds.open("seedsout");
    GPU_ASSERT(cudaMallocManaged(&buffer, sizeof(*buffer) * SEEDS_PER_CALL));
    GPU_ASSERT(cudaPeekAtLastError());
    GPU_ASSERT(cudaMallocManaged(&counter, sizeof(*counter)));
    GPU_ASSERT(cudaPeekAtLastError());
    time_t start_time = time(NULL);
    int outCount = 0;

    int checkpointTemp = 0;
    for(uint64_t offset =offsetStart;offset<seedCount;offset+=SEEDS_PER_CALL) {
        // Normal filtering
        time_t elapsed = time(NULL) - start_time;
        double frac = (double) offset / (double)(seedCount);
        #ifdef BOINC
            boinc_fraction_done(frac);
        #endif
        *counter = 0;
        tempCheck<<<1ULL<<WORK_SIZE_BITS,BLOCK_SIZE>>>(START + offset, buffer,counter);
        GPU_ASSERT(cudaPeekAtLastError());
        GPU_ASSERT(cudaDeviceSynchronize());  

        for(int i=0;i<*counter;i++) {
            filtered_once.push_back(buffer[i]);
        }

        //std::cout << "2nd level candidates: " << filtered_twice.size() << std::endl;
		
        if (filtered_once.size() >= SEEDS_PER_CALL || offset + SEEDS_PER_CALL >= seedCount) {
            //2nd level of filtering
            std::cout << "2nd level filtering" << std::endl;
            int toCopy = filtered_once.size() > SEEDS_PER_CALL ? SEEDS_PER_CALL : filtered_once.size();

            for(int i=0;i<toCopy;i++) {
                buffer[i] = filtered_once.back();
                //std::cout << "TRY: " << buffer[i] << std::endl;
                filtered_once.pop_back();
            }

            tempCheck2<<<(toCopy/BLOCK_SIZE)+1,BLOCK_SIZE>>>(toCopy, buffer);
            GPU_ASSERT(cudaPeekAtLastError());
            GPU_ASSERT(cudaDeviceSynchronize());

            for(int i=0;i<toCopy;i++) {
                if (buffer[i]!=0) {
                    uint64_t seed = buffer[i];
                    //std::cout << "2nd level seed found: " << seed << std::endl;
                    filtered_twice.push_back(seed);
                    //outSeeds << seed << std::endl;
                    //outCount++;
                }
            }
        }

        if (filtered_twice.size() >= SEEDS_PER_CALL || offset + SEEDS_PER_CALL >= seedCount) {
            //2nd level of filtering
            std::cout << "3rd level filtering" << std::endl;
            int toCopy = filtered_twice.size() > SEEDS_PER_CALL ? SEEDS_PER_CALL : filtered_twice.size();

            for(int i=0;i<toCopy;i++) {
                buffer[i] = filtered_twice.back();
                //std::cout << "TRY: " << buffer[i] << std::endl;
                filtered_twice.pop_back();
            }

            tempCheck3<<<(toCopy/BLOCK_SIZE)+1,BLOCK_SIZE>>>(toCopy, buffer);
            GPU_ASSERT(cudaPeekAtLastError());
            GPU_ASSERT(cudaDeviceSynchronize());

            for(int i=0;i<toCopy;i++) {
                if (buffer[i]!=0) {
                    uint64_t seed = buffer[i];
                    std::cout << "3rd level seed found: " << seed << std::endl;
                    outSeeds << seed << std::endl;
                    outCount++;
                }
            }
        }
		
        if(checkpointTemp >= 180000000 || boinc_time_to_checkpoint()){
            #ifdef BOINC
		        boinc_begin_critical_section(); // Boinc should not interrupt this
            #endif
            // Checkpointing section below
			boinc_delete_file("packpoint.txt"); // Don't touch, same func as normal fdel
            FILE *checkpoint_data = boinc_fopen("packpoint.txt", "wb");
            std::ofstream firstvector_data;
            std::ofstream secondvector_data;
            firstvector_data.open("vector1.txt");
            secondvector_data.open("vector2.txt");
			struct checkpoint_vars data_store;
			data_store.offset = offset;
			data_store.elapsed_chkpoint = elapsed_chkpoint + elapsed;
            data_store.vector1Size = filtered_once.size();
            data_store.vector2Size = filtered_twice.size();
            for(uint64_t i = 0; i < filtered_once.size(); i++){
                firstvector_data << filtered_once[i] << std::endl;
            }
            for(uint64_t i = 0; i < filtered_twice.size(); i++){
                secondvector_data << filtered_twice[i] << std::endl;
            }
            fwrite(&data_store, sizeof(data_store), 1, checkpoint_data);

            fclose(checkpoint_data);
            firstvector_data.close();
            secondvector_data.close();
            checkpointTemp = 0;
            #ifdef BOINC
            boinc_end_critical_section();
            boinc_checkpoint_completed(); // Checkpointing completed
            #endif
        }
        checkpointTemp += SEEDS_PER_CALL;
        std::cout << "Seeds left:" << (((int64_t)seedCount-offset)-SEEDS_PER_CALL) << std::endl;  
    }

    std::cout << "Done processing" << std::endl;    
    #ifdef BOINC
	    boinc_begin_critical_section();
	#endif
    time_t elapsed = time(NULL) - start_time;
    double done = (double)COUNT / 1000000.0;
    double speed = done / (double) elapsed;
    fprintf(stderr, "\nSpeed: %.2lfm/s\n", speed );
    fprintf(stderr, "Done\n");
    fprintf(stderr, "Processed: %llu seeds in %.2lfs seconds\n", COUNT, (double) elapsed_chkpoint + (double) elapsed );
    fprintf(stderr, "Have %llu output seeds.\n", outCount);
    fflush(stderr);
    outSeeds.close();
    boinc_delete_file("packpoint.txt");
    #ifdef BOINC
        boinc_end_critical_section();
    #endif
    boinc_finish(0);
}

double getNextDoubleForLocNoise(int x, int z) {
    Random rand = get_random((((int64_t)x) >> 4) * 341873128712LL + (((int64_t)z) >> 4) * 132897987541LL);
    for (int dx = 0; dx < 16; dx++) {
      for (int dz = 0; dz < 16; dz++) {
        if (dx == (x & 15) && dz == (z & 15)) {
          //advance2(&rand);
          //advance2(&rand);
          return next_double(&rand);
        }
        advance2(&rand);
        advance2(&rand);
        advance2(&rand);
        for(int k1 = 127; k1 >= 0; k1--) {
          random_next_int_nonpow(&rand,5);
        }
        //for (int i = 0; i < 67; i++) {
        //  advance2(&rand);
        //}
      }
    }
    exit(-99);
}
