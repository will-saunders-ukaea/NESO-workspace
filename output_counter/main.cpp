#include <CL/sycl.hpp>
using namespace cl;

#include <type_traits>
#include <vector>
#include <cstdlib>
#include <memory>
#include <list>
#include <iostream>
#include <vector>


struct OutputCounter {

  /*
   * These members are scalar types which are trivially copyable (unlike shared
   * pointers). This does force the class to hold a pointer to the device which
   * in turn requires care to ensure use-after-free does not occur.
   */
  sycl::queue * queue;
  std::size_t N;
  /*
   * Following CUDA conventions this is prefixed with a d_ to denote device
   */
  std::size_t * d_ptr;
  
  /**
   *  Constructor, host callable only.
   */
  OutputCounter(
    sycl::queue &queue,
    const size_t N
  ) : queue(&queue), N(N) {
    this->d_ptr = static_cast<size_t *>(
        sycl::malloc_device(N * sizeof(size_t), *this->queue));
  };
  
  /**
   *  This is a user called function (host) as if it were in the destructor:
   *  1) The user provided destructor makes the class non-trivially copyable.
   *  2) More logic would be needed to avoid double frees on the pointer when
   *     the obejct is copied.
   */
  inline void free(){
    if (this->d_ptr != nullptr){
      sycl::free(this->d_ptr, *this->queue);
      this->d_ptr = nullptr;
    }
  };
  
  /* 
   * This is a host callable function and not device callable. This zeros the
   * counts in the underlying buffer - which is a common buffer across all
   * copies of the object.
   */
  inline void pre_kernel(){
    // could just do a memcpy here
    auto k_ptr = this->d_ptr;
    queue->submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(N), [=](sycl::id<1> idx) {
                k_ptr[idx] = 0;
              });
        })
        .wait_and_throw();
  }

  /* 
   * Host callable and not device callable. Gets the counter values.
   */
  inline std::vector<size_t> get_counts() const {
    
    std::vector<size_t> host_counts(this->N);
    sycl::buffer<size_t, 1> b_counts(host_counts.data(), host_counts.size());
    auto k_ptr = this->d_ptr;
    queue->submit([&](sycl::handler &cgh) {
          auto a_counts = b_counts.get_access<sycl::access_mode::write>(cgh);
          cgh.parallel_for<>(
              sycl::range<1>(N), [=](sycl::id<1> idx) {
                a_counts[idx] = k_ptr[idx];
              });
        })
        .wait_and_throw();
    return host_counts;
  }

  /*
   * This is a device callable function and not host callable.
   * It returns a unique "position" of the caller for the index and atomically
   * increments the counter.
   */
  inline size_t get_add_output(
    const int index
  ) const {
     sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                      sycl::memory_scope::device>
         t(d_ptr[index]);
     return t.fetch_add(size_t(1));
  }

};


int main(int argc, char** argv){

  static_assert(std::is_trivially_copyable_v<OutputCounter> == true, 
      "OutputCounter is not trivially copyable to device");

  sycl::device device = sycl::device(sycl::default_selector());
  std::cout << "Using " << device.get_info<sycl::info::device::name>()
    << std::endl;
  sycl::queue queue = sycl::queue(device);
  OutputCounter output_counter_4(queue, 4);

  const std::size_t N = 1024;
  output_counter_4.pre_kernel();

  queue.submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(N), [=](sycl::id<1> idx) {
                
                // Each thread increments the mod(idx, 4) counter.
                // Index is the value of the counter before it was incremented.
                const size_t index = output_counter_4.get_add_output(idx % 4);

              });
        })
        .wait_and_throw();
  
  // print the counts
  auto counts = output_counter_4.get_counts();
  for(auto cx : counts){
    std::cout << cx << std::endl;
  }
  
  // free the underlying buffer in the counter
  output_counter_4.free();

  return 0;
}

