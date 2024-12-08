#define MIPP_ALIGNED_LOADS

#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using element_type = float;

static const element_type pi = 4*atan(1);
static const int DEFAULT_NX = 10;

template <typename T>
__attribute__((noinline)) static void box_muller(std::vector<T> &z0, std::vector<T> &z1, const std::vector<T> &u1, const std::vector<T> &u2)
{
        const std::size_t nx = u1.size();
        assert(u2.size() == nx);
        assert(z0.size() == nx);
        assert(z1.size() == nx);

//#pragma omp parallel for simd schedule(static) shared(z1, u1, u2)
        for (std::size_t i = 0; i < nx; ++i)
        {
                z0[i] = sqrt(log(u1[i])*-2) * cos(u2[i]*2*pi);
                z1[i] = sqrt(log(u1[i])*-2) * sin(u2[i]*2*pi);
        }
}

int main(int argc, char *argv[])
{
        std::size_t nx = DEFAULT_NX;

        int argi = 1;
        while (argi < argc)
        {
                if (argi == 1)
                {
                        nx = std::stoi(argv[argi]);
                }
                else
                {
                        std::cerr << "invalid number of arguments" << std::endl;
                        exit(EXIT_FAILURE);
                }

                ++argi;
        }

        std::vector<element_type> u1(nx);
        std::vector<element_type> u2(nx);

        std::vector<element_type> z0(nx);
        std::vector<element_type> z1(nx);

        {
                std::uniform_real_distribution<element_type> random_distribution(0, 1);
                auto random_generator = std::bind(random_distribution, std::minstd_rand());

//#pragma omp parallel for schedule(static) shared(z1, u1, u2)
                for (std::size_t i=0; i<nx; i++)
                {
                        u1[i] = random_generator();
                        u2[i] = random_generator();
                        z0[i] = 0;
                        z1[i] = 0;
                }
        }

        box_muller(z0, z1, u1, u2);

        return 0;
}
