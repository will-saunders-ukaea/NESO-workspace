#ifndef _EXPONENTIAL_DENSITY
#define _EXPONENTIAL_DENSITY

#include <tgmath.h>
#include <random>
#include <vector>

using namespace std;


inline double interval_to_ref(const double x){
  return (x - 0.5) * 2.0;
}

inline double ref_to_interval(const double x){
  return (x + 1.0) * 0.5;
}

inline double cumulative_exp(const double x){
  const double sqrt_two_reciprocal = 0.7071067811865475;
  return 0.5 * (1.0 + erf(x * sqrt_two_reciprocal));
}

inline double get_density_left(){
  return erf(interval_to_ref(0.0));
}

inline double get_density_right(){
  return erf(interval_to_ref(1.0));
}



/*
 // No erfinv in C++ std library
inline double sample_unit_interval_gaussian(std::mt19937 &rng){
  std::uniform_real_distribution<double> point_distribution(
    get_density_left(), 
    get_density_right()
  );
  const double y_sample = point_distribution(rng);
  
  return ref_to_interval(erfinv(y_sample));
}
*/

template <typename T, typename U>
inline void monte_carlo_sample_2d(T &target_function, std::vector<U> &coords, std::mt19937 &rng){



}









#endif
