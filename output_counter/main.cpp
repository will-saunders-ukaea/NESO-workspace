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
   * These members are simple types which are trivially copyable (unlike shared
   * pointers).
   */
  sycl::queue * queue;
  std::size_t N;
  // Following CUDA conventions this is prefixed with a d_ to denote device
  // memory.
  std::size_t * d_ptr;
  
  // This is a host callable function and not device callable
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

  // host callable and not device callable
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

  // This is a device callable function and not host callable.
  // It returns a unique "position" of the caller for the index.
  inline size_t get_add_output(
    const int index
  ) const {
     sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                      sycl::memory_scope::device>
         t(d_ptr[index]);
     return t.fetch_add(size_t(1));
  }

};

/**
 *  Slightly overcomplicated class to malloc and free the OutputContainers
 */
struct OutputCounterFactory {
  std::shared_ptr<sycl::queue> queue;
  std::list<size_t*> alloced;
  OutputCounterFactory(std::shared_ptr<sycl::queue> queue) : queue(queue) {};

  // on destruction of this object free all the device memory which was allocated by the instance
  ~OutputCounterFactory() {
    for(size_t * ptr : this->alloced){
      sycl::free(ptr, *this->queue);
    }
  };
  
  inline OutputCounter create(const size_t N){
    OutputCounter oc;
    oc.queue = this->queue.get();
    oc.N = N;
    size_t * ptr = static_cast<size_t *>(sycl::malloc_device(N * sizeof(size_t), *this->queue));
    oc.d_ptr = ptr;
    this->alloced.push_back(ptr);
    return oc;
  }
};


int main(int argc, char** argv){

  static_assert(std::is_trivially_copyable_v<OutputCounter> == true, 
      "OutputCounter is not trivially copyable to device");


  sycl::device device = sycl::device(sycl::default_selector());
  auto queue = std::make_shared<sycl::queue>(device);
  auto output_counter_factory = std::make_shared<OutputCounterFactory>(queue);
  auto output_counter_4 = output_counter_factory->create(4);

  const std::size_t N = 1024;
  output_counter_4.pre_kernel();

  queue->submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(N), [=](sycl::id<1> idx) {
                output_counter_4.get_add_output(idx % 4);
              });
        })
        .wait_and_throw();

  auto counts = output_counter_4.get_counts();
  for(auto cx : counts){
    std::cout << cx << std::endl;
  }

  return 0;
}

