#include <string>
#include <iostream>
#include <RAJA/RAJA.hpp>
void SpMV_dispatch(int dimSize, double nonzeroDensity, 
                   int runDense, int runSpecialized, int runSparseRAJA);


void usage() {
  std::cerr << "Usage: ./sparseEval.exe RunDense RunSpecialized RunSparseRAJA RunSpMV RunGauSei RunInCholFact DimSize NonzeroDensity\n"; 
}

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

    std::cout << "SpMV,Dense," << dimSize << "," << nonzeroDensity << "," << elapsed << ",1.0\n";
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
    std::cout << "SpMV,Specialized," << dimSize << "," << nonzeroDensity << "," << elapsed << ",1.0\n";

    delete[] A_cols.get_data();
    delete[] A_rows.get_data();
    delete[] A_vals.get_data();
    delete[] y.get_data();
  } // SpMV runSpecialized

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

    auto knl = make_sparse_kernel_sym<POLICY>(make_tuple(0,0), dense_segs, refData, lam);

    auto start = now();
    for(int i = 0; i < numReps; i++) {
      knl();
    }
    auto stop = now();
    auto elapsed = elapsed_time(start, stop);
    std::cout << "SpMV,SparseRAJA," << dimSize << "," << nonzeroDensity << "," << elapsed << "," << refData.get_hit_rate() << "\n";

    std::cerr << "Hit rate: " << refData.get_hit_rate() << "\n";
    delete[] y.get_data();
  } // SpMV runSparseRAJA

  delete[] x.get_data();
} // SpMV_dispatch




void GauSei_dispatch(int dimSize, double nonzeroDensity, 
                   int runDense, int runSpecialized, int runSparseRAJA) {
  int numReps = 1000;
  auto refData = make_random_sparse_view2<double,true,false>(dimSize, nonzeroDensity);
  DenseView1 b(new double[dimSize], dimSize);
  for(int i = 0; i < dimSize; i++) {
    b(i) = std::rand(); 
  }
 
  for(int i = 0; i < dimSize; i++) {
    if (refData(i,i) == 0) {
      std::cerr << "GauSei has a zero on the diagonal" << i << "\n";
      return;
    }
  } 
  if(runDense) {
    DenseView2 A(new double[dimSize*dimSize], dimSize, dimSize);
    DenseView1 x(new double[dimSize], dimSize);
    double temp;
 
    //copy in and setup data
    std::memset(A.get_data(), 0, dimSize*dimSize);
    std::memset(x.get_data(), 0, dimSize);
    for(int i = 0; i < refData.size(); i++) {
      auto i0 = refData.dim(0)[i];
      auto i1 = refData.dim(0)[i];
      A(i0,i1) = refData(i0,i1);
    }
        
    using POLICY = KernelPolicy<
      statement::For<0,loop_exec,
        statement::Lambda<0,Segs<0>>,
        statement::For<1,loop_exec,
          statement::Lambda<1>
        >,
        statement::Lambda<2,Segs<0>>
      >
    >;

    auto lam1 = [&](auto i) {
      temp = 0.0;
    };
    auto lam2 = [&](auto i, auto j) {
      if (j != i) {
        temp += A(i,j) * x(j);
      }
    };
    auto lam3 = [&](auto i) {
      x(i) = (b(i) - temp) / A(i,i);
    };

    auto seg1 = RangeSegment(0,dimSize);
    auto seg2 = RangeSegment(0,dimSize);
    auto segs = make_tuple(seg1, seg2);

    auto knl = make_kernel<POLICY>(segs, lam1, lam2, lam3);

    auto start = now();
    for(int i = 0; i < numReps; i++) {
      knl();
    }
    auto stop = now();
    auto elapsed = elapsed_time(start, stop);
    std::cout << "GauSei,Dense," << dimSize << "," << nonzeroDensity << "," << elapsed << ",1.0\n";


  } //GauSei Dense

  if (runSpecialized) {
    size_t numNonZeros = refData.size();
    DenseView1 x(new double[dimSize], dimSize);
    DenseView1 A_rows(new double[numNonZeros], numNonZeros);
    DenseView1 A_cols(new double[numNonZeros], numNonZeros);
    DenseView1 A_vals(new double[numNonZeros], numNonZeros);

    //set and copy data
    std::memset(x.get_data(), 0, dimSize);
    for(int i = 0; i < numNonZeros; i++) {
      A_rows(i) = refData.dim(0)[i];
      A_cols(i) = refData.dim(1)[i];
      A_vals(i) = refData.val(i);
    }

    auto seg = RangeSegment(0,numNonZeros);

    int prev_i = 0;
    double temp = 0.0;
    auto lam = [&](auto idx) {
      int i = A_rows(idx);
      int j = A_cols(idx);
      double v = A_vals(idx);

      if (i != prev_i) {
        double prev_diagonal = refData(prev_i, prev_i);
        x(prev_i) = (b(prev_i) - temp) / prev_diagonal;
        temp = 0.0;
        prev_i = i;
      }
      if (j != i) {
        temp += v * x(j);
      } 
    };

    auto knl = make_forall<loop_exec>(seg, lam);

    auto start = now();
    for(int i = 0; i < numReps; i++) {
      knl();
      auto prev_diagonal = refData(prev_i, prev_i);
      x(prev_i) = (b(prev_i) - temp) / prev_diagonal;
    }
    auto stop = now();
    auto elapsed = elapsed_time(start, stop);
    std::cout << "GauSei,Specialized," << dimSize << "," << nonzeroDensity << "," << elapsed <<",1.0" << "\n";

    delete[] A_rows.get_data();
    delete[] A_cols.get_data();
    delete[] A_vals.get_data();
    delete[] x.get_data();

  } //GauSei Specialized

  if (runSparseRAJA) {
    auto A = make_sparse_view_diag<double>(refData.impl.indices[0], refData.impl.indices[1], refData.impl.val);
    refData.reset_counters();
    DenseView1 x(new double[dimSize], dimSize);
    double temp = 0.0;
    using POLICY = KernelPolicy<
      statement::For<0,loop_exec,
        statement::Lambda<0,Segs<0>>,
        statement::For<1,loop_exec,
          statement::Lambda<1>
        >,
        statement::Lambda<2,Segs<0>>
      >
    >;


    auto seg1 = RangeSegment(0,dimSize);
    auto seg2 = RangeSegment(0,dimSize);
    auto dense_segs = make_tuple(seg1, seg2);


    auto lam1 = [&](auto i) {
      temp = 0.0;
    };
    auto lam2 = [&](auto i, auto j) { 
      if (j != i) {
        auto summand = A(i,j) * x(j);
        temp += summand;
      }
    };
    auto lam3 = [&](auto i) {
      x(i) = (b(i) - temp) / A(i,i);
    };

    auto knl = make_sparse_kernel_sym<POLICY, 1>(make_tuple(0,1), dense_segs, A, lam1, lam2, lam3);
    auto start = now();
    for(int i = 0; i < numReps; i++) {
      knl();
    }
    auto stop = now();
    auto elapsed = elapsed_time(start, stop);
    std::cout << "GauSei,SparseRAJA," << dimSize << "," << nonzeroDensity << "," << elapsed << "," << A.get_hit_rate() << "\n";

    std::cerr << "Hit rate: " << A.get_hit_rate() << "\n";
    std::cerr << "Num hits: " << A.get_hits() << "\n";
    std::cerr << "Num misses: " << A.get_misses() << "\n";

    delete[] x.get_data();


  } //GauSei SparseRAJA

  delete[] b.get_data();

} //GauSei



int main(int argc, char * argv[]) {
  
  if (argc != 9) {
    std::cerr << "Error: Please provide 8 arguments.\n";
    usage();
    return 1;
  }
  
  int runDense = std::atoi(argv[1]);
  int runSpecialized = std::atoi(argv[2]);
  int runSparseRAJA = std::atoi(argv[3]);
  
  int runSpMV = std::atoi(argv[4]);
  int runGauSei = std::atoi(argv[5]);
  int runInCholFact = std::atoi(argv[6]);
  
  int dimSize = std::atoi(argv[7]);
  if (dimSize < 1) {
    std::cerr << "Error: dimSize should positive.\n";
    usage();
    return 2;
  }
  
  double nonzeroDensity = std::stod(argv[8]);
  if (nonzeroDensity <= 0 || nonzeroDensity > 1) {
    std::cerr << "Error: Nonzero density should be between 0 and 1.\n";
    usage();
    return 3;
  }
  
  if(runSpMV) {
    SpMV_dispatch(dimSize, nonzeroDensity, runDense, runSpecialized, runSparseRAJA);
  }
  
  if(runGauSei) {
    GauSei_dispatch(dimSize, nonzeroDensity, runDense, runSpecialized, runSparseRAJA);
  }
  
  if(runInCholFact) {
    //InCholFact_dispatch(dimSize, nonzeroDensity, runDense, runSpecialized, runSparseRAJA);
  }

  return 0; 
}

