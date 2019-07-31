#include <array>
#include <iostream>
#include <random>

#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int argc, char** argv)
{
    sycl::device device = sycl::default_selector {}.select_device();
    sycl::queue queue(device, [](sycl::exception_list el) {
        for (auto ex : el) {
            std::rethrow_exception(ex);
        }
    });

    // set work group size to 32
    size_t wgroup_size = 32;

    auto part_size = wgroup_size * 2;

    auto has_loc_mem = device.is_host()
        || (device.get_info<sycl::info::device::local_mem_type>()
            != sycl::info::local_mem_type::none);

    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

    if (!has_loc_mem || local_mem_size < (wgroup_size * sizeof(int32_t))) {
        throw "Device doesn't have enough local memory!";
    }

    std::cout << "We've set up the SYCL queue" << std::endl;
    std::cout << "Device has local memory: " << (has_loc_mem ? "True" : "False") << std::endl;
    std::cout << "Local memory size: " << local_mem_size << std::endl;

    return 0;
}