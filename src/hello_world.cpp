#include <iostream>

#include <CL/sycl.hpp>

class vector_addition;

int main(int argc, char** argv)
{
    // create vectors
    cl::sycl::float4 a = { 1.0, 2.0, 3.0, 4.0 };
    cl::sycl::float4 b = { 4.0, 3.0, 2.0, 1.0 };
    cl::sycl::float4 c = { 0.0, 0.0, 0.0, 0.0 };

    // use default device selector
    cl::sycl::default_selector selector;
    // create sycl queue
    cl::sycl::queue queue(selector);

    // get device info
    std::cout << "Running on "
              << queue.get_device().get_info<cl::sycl::info::device::name>()
              << "\n\n";

    /* set up buffers
   NOTE: buffers memory passed to the buffers *cannot* be used
   while the buffers exist */
    {
        cl::sycl::buffer<cl::sycl::float4, 1> a_sycl(&a, cl::sycl::range<1>(1));
        cl::sycl::buffer<cl::sycl::float4, 1> b_sycl(&b, cl::sycl::range<1>(1));
        cl::sycl::buffer<cl::sycl::float4, 1> c_sycl(&c, cl::sycl::range<1>(1));

        // create command  group
        queue.submit([&](cl::sycl::handler& cgh) {
            // set up accessors so that the vectors can be used
            auto a_acc = a_sycl.get_access<cl::sycl::access::mode::read>(cgh);
            auto b_acc = b_sycl.get_access<cl::sycl::access::mode::read>(cgh);
            auto c_acc = c_sycl.get_access<cl::sycl::access::mode::discard_write>(cgh);

            // execute exactly once on the device
            cgh.single_task<class vector_addition>(
                [=]() { c_acc[0] = a_acc[0] + b_acc[0]; });
        });
    }
    /* buffers are destroyed and it is safe to use
     the vectors again */

    // show results
    std::cout << "  A { " << a.x() << ", " << a.y() << ", " << a.z() << ", "
              << a.w() << " }\n"
              << "+ B { " << b.x() << ", " << b.y() << ", " << b.z() << ", "
              << b.w() << " }\n"
              << "------------------\n"
              << "= C { " << c.x() << ", " << c.y() << ", " << c.z() << ", "
              << c.w() << " }" << std::endl;

    return 0;
}