#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>

#include <CL/sycl.hpp>

class reduction_kernel;
namespace sycl = cl::sycl;

int main(int, char**)
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

    sycl::buffer<int32_t, 1> buf(arr.data(), sycl::range<1>(arr.size()));

    sycl::device device = sycl::default_selector {}.select_device();

    sycl::queue queue(device, [](sycl::exception_list el) {
        for (auto ex : el) {
            std::rethrow_exception(ex);
        }
    });

    auto wgroup_size = 32;
    auto part_size = wgroup_size * 2;

    auto has_local_mem = device.is_host() || (device.get_info<sycl::info::device::local_mem_type>() != sycl::info::local_mem_type::none);
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    if (!has_local_mem || local_mem_size < (wgroup_size * sizeof(int32_t))) {
        throw "Device doesn't have enough local memory!";
    }

    std::cout << "Device has local memory: " << (has_local_mem ? "True" : "False")
              << std::endl;
    std::cout << "Local memory size: " << local_mem_size << std::endl;

    // reduction loop
    auto len = arr.size();
    while (len != 1) {
        auto n_wgroups = (len + part_size - 1) / part_size;

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                sycl::access::target::local>
                local_mem(sycl::range<1>(wgroup_size), cgh);

            auto global_mem = buf.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class reduction_kernel>(
                sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
                [=](sycl::nd_item<1> item) {
                    // get local and global ids
                    size_t local_id = item.get_local_linear_id();
                    size_t global_id = item.get_global_linear_id();
                    // zero-initialise local memory
                    // as it may contain garbage data
                    local_mem[local_id] = 0;

                    if ((2 * global_id) < len) {
                        local_mem[local_id] = global_mem[2 * global_id] + global_mem[2 * global_id + 1];
                    }
                    // synchronise
                    item.barrier(sycl::access::fence_space::local_space);

                    // reduce each work-group's array in local memory
                    for (size_t stride = 1; stride < wgroup_size; stride *= 2) {
                        auto idx = 2 * stride * local_id;
                        if (idx < wgroup_size) {
                            local_mem[idx] = local_mem[idx] + local_mem[idx + stride];
                        }
                        // synchronise since each iteration depends on
                        // the results of the previous one
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (local_id == 0) {
                        // write the result to global memory
                        global_mem[item.get_group_linear_id()] = local_mem[0];
                    }
                });
        });
        queue.wait_and_throw();

        len = n_wgroups;
    }

    auto acc = buf.get_access<sycl::access::mode::read>();

    int32_t expected = 0;
    for (auto e : arr)
        expected += e;

    auto actual = acc[0];

    std::cout << std::endl;
    std::cout << "Expected result: " << expected << std::endl;
    std::cout << "Actual result: " << actual << std::endl;
    std::cout << "Expected == Actual: " << (expected == actual ? "True" : "False")
              << std::endl;

    return 0;
}