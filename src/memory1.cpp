#include <array>
#include <iostream>
#include <random>

#include <CL/sycl.hpp>

int main(int argc, char** argv)
{
    std::array<int32_t, 16> arr;
    std::mt19937 mt_engine(std::random_device {}());
    std::uniform_int_distribution<int32_t> idist(0, 10);

    std::cout << "Data: ";
    for (auto& el : arr) {
        el = idist(mt_engine);
        std::cout << el << " ";
    }
    std::cout << std::endl;

    cl::sycl::buffer<int32_t, 1> buf(arr.data(), cl::sycl::range<1>(arr.size()));

    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    std::cout << "The SYCL buffer is set up" << std::endl;

    return 0;
}