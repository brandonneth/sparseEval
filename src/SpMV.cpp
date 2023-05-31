#include <RAJA/RAJA.hpp>
#include <chrono>

using namespace RAJA;
using DenseView1 = View<double, Layout<1>>;
using DenseView2 = View<double, Layout<2>>;

std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now(); 
}
auto elapsed_time(std::chrono::time_point<std::chrono::high_resolution_clock> start, std::chrono::time_point<std::chrono::high_resolution_clock> stop) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}
void SpMV_dispatch(int dimSize, double nonzeroDensity, 
                   int runDense, int runSpecialized, int runSparseRAJA) {
  int numReps = 1000;
  
  auto refData = make_random_sparse_view2<double>(dimSize, nonzeroDensity);
  DenseView1 x(new double[dimSize], dimSize);
  for(int i = 0; i < dimSize; i++) {
    x(i) = std::rand(); 
  }
  
  if(runDense) {
    DenseView2 A(new double[dimSize*dimSize], dimSize, dimSize);
    DenseView1 y(new double[dimSize], dimSize);
    
    //copy in and setup data
    std::memset(A.get_data(), 0, dimSize*dimSize);
    std::memset(y.get_data(), 0, dimSize);
    for(int i = 0; i < dimSize; i++) {
      for(size_t j = 0; j < dimSize; j++) {
        A(i,j) = refData(i,j);
      }
    }
    
    using POLICY = KernelPolicy<
      statement::For<0,loop_exec,
        statement::For<1,loop_exec,
          statement::Lambda<0>
        >
      >
    >;

    auto seg1 = RangeSegment(0,dimSize);
    auto seg2 = RangeSegment(0,dimSize);
    auto segs = make_tuple(seg1, seg2);

    auto lam = [&](auto i, auto j) {
      y(i) += A(i,j) * x(j);
    };

    auto knl = make_kernel<POLICY>(segs, lam);
    
    std::chrono::time_point<std::chrono::high_resolution_clock> start = now();
    for(int i = 0; i < numReps; i++) {
      knl();
    }
    auto stop = now();
    
    auto elapsed = elapsed_time(start, stop);
    
    std::cout << "SpMV,Dense," << dimSize << "," << nonzeroDensity << "," << elapsed << "\n";
    delete[] A.get_data();
    delete[] y.get_data();
  } // runDense
  
  if(runSpecialized) {
    size_t numNonZeros = refData.size();
    DenseView1 y(new double[dimSize], dimSize);
    DenseView1 A_rows(new double[numNonZeros], numNonZeros);
    DenseView1 A_cols(new double[numNonZeros], numNonZeros);
    DenseView1 A_vals(new double[numNonZeros], numNonZeros);
    
    //set and copy data
    std::memset(y.get_data(), 0, dimSize);
    for(int i = 0; i < numNonZeros; i++) {
      A_rows(i) = refData.dim(0)[i];
      A_cols(i) = refData.dim(1)[i];
      A_vals(i) = refData.val(i);
    }
    
    auto seg = RangeSegment(0,numNonZeros);

    auto lam = [&](auto idx) {
      auto i = A_rows(idx);
      auto j = A_cols(idx);
      y(i) += A_vals(idx) * x(j);
    };

    auto knl = make_forall<loop_exec>(seg, lam);

    auto start = now();
    for(int i = 0; i < numReps; i++) {
      knl(); 
    }
    auto stop = now();
    auto elapsed = elapsed_time(start, stop);
    std::cout << "SpMV,Specialized," << dimSize << "," << nonzeroDensity << "," << elapsed << "\n";
    
    delete[] A_cols.get_data();
    delete[] A_rows.get_data();
    delete[] A_vals.get_data();
    delete[] y.get_data();
  } // runSpecialized
  
  if(runSparseRAJA) {
    refData.reset_counters();
    DenseView1 y(new double[dimSize], dimSize);
    
    using POLICY = KernelPolicy<
      statement::For<0,loop_exec,
        statement::For<1,loop_exec,
          statement::Lambda<0>
        >
      >
    >;

    auto seg1 = RangeSegment(0,dimSize);
    auto seg2 = RangeSegment(0,dimSize);
    auto dense_segs = make_tuple(seg1, seg2);

     
    auto lam = [&](auto i, auto j) {
      y(i) += refData(i,j) * x(j);
    };
    
    auto knl = make_sparse_kernel<POLICY>(dense_segs, refData, lam);
  
    auto start = now();
    for(int i = 0; i < numReps; i++) {
      knl(); 
    }
    auto stop = now();
    auto elapsed = elapsed_time(start, stop);
    std::cout << "SpMV,SparseRAJA," << dimSize << "," << nonzeroDensity << "," << elapsed << "\n";
    
    std::cerr << "Hit rate: " << refData.get_hit_rate() << "\n"; 
    delete[] y.get_data();
  } // runSparseRAJA
  
  delete[] x.get_data();
} // SpMV_dispatch

  
