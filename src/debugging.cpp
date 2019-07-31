#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

class printkernel;

int main(int argc, char** argv)
{
    sycl::queue queue(sycl::default_selector {});

    queue.submit([&](sycl::handler& cgh) {
        sycl::stream out(1024, 256, cgh);

        cgh.single_task<class printkernel>([=] {
            out << "Hello stream!" << sycl::endl;
        });
    });

    queue.wait_and_throw();

    return 0;
}