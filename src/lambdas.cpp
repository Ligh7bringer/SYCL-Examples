#include <iostream>
#include <numeric>
#include <vector>

#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

namespace kernels {
class generator_kernel {
};
}

template <typename T, class Operation>
void generate_seq(const std::vector<T>& input, std::vector<T>& output, size_t num_elements, Operation op)
{
    sycl::buffer<T, 1> input_buf(input.data(), sycl::range<1>(num_elements));
    sycl::buffer<T, 1> output_buf(output.data(), sycl::range<1>(num_elements));
    sycl::queue queue(sycl::default_selector {});
    queue.submit([&](sycl::handler& cgh) {
        auto input_acc = input_buf.template get_access<sycl::access::mode::read>(cgh);
        auto output_acc = output_buf.template get_access<sycl::access::mode::write>(cgh);

        // Execute kernel
        cgh.parallel_for<kernels::generator_kernel>(
            sycl::range<1>(num_elements), [=](sycl::id<1> idx) mutable {
                output_acc[idx] = op(input_acc[idx]);
            });
    });
}

int main(int argc, char** argv)
{
    constexpr size_t num_elements = 50000;
    std::vector<unsigned long> input(num_elements);
    std::iota(input.begin(), input.end(), 0);
    std::vector<unsigned long> output(num_elements);
    std::fill(output.begin(), output.end(), 0);

    auto increment = [res = 0](auto idx) mutable {
        for (size_t i = 0; i < idx; i++) {
            res++;
        }

        return res;
    };

    auto increment_and_multiply_by_scalar =
        [increment, res = 0, scalar = 10](auto idx) mutable {
            res = increment(idx);
            res *= scalar;
            return res;
        };

    auto fibonacci = [res = 0, a = 1, b = 1](auto idx) mutable {
        if (idx <= 0)
            return 0;

        if (idx > 0 && idx < 3)
            return 1;

        for (size_t i = 2; i < idx; i++) {
            res = a + b;
            a = b;
            b = res;
        }
        return res;
    };

    generate_seq(input, output, num_elements, increment);

    for (auto el : output)
        std::cout << el << std::endl;

    return 0;
}