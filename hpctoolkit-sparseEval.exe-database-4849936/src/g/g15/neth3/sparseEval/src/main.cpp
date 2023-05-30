#include <string>
#include <iostream>


void SpMV_dispatch(int dimSize, double nonzeroDensity, 
                   int runDense, int runSpecialized, int runSparseRAJA);


void usage() {
  std::cerr << "Usage: ./sparseEval.exe RunDense RunSpecialized RunSparseRAJA RunSpMV RunGauSei RunInCholFact DimSize NonzeroDensity\n"; 
}
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
  if (dimSize < 1 || dimSize > 2000) {
    std::cerr << "Error: dimSize should be between 1 and 20.\n";
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
    //GauSei_dispatch(dimSize, nonzeroDensity, runDense, runSpecialized, runSparseRAJA);
  }
  
  if(runInCholFact) {
    //InCholFact_dispatch(dimSize, nonzeroDensity, runDense, runSpecialized, runSparseRAJA);
  }

  return 0; 
}
