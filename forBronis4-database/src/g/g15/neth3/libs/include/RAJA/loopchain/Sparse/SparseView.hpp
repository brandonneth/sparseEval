// Contains functions that extract isl polyhedral sets from kernel objects.

#ifndef RAJA_sparse_view_HPP
#define RAJA_sparse_view_HPP
#include <cassert>
#include "RAJA/loopchain/Sparse/SparseImpl.hpp"
#include "RAJA/loopchain/Sparse/COO.hpp"
#include "RAJA/loopchain/Sparse/DIAG.hpp"

namespace RAJA {

template <camp::idx_t NumDims>
struct SparseSymbolicSegmentLead;

template <camp::idx_t NumDims>
struct SparseSymbolicSegmentFollow;

template <typename ElmType, size_t NumDims, typename FormatType>
struct  SparseView {
 
 
  FormatType impl;

  SparseView(FormatType i) : impl(i) {}

  void* get_data() { return impl.get_data();}
  
  ElmType & val(idx_t i) {return impl.val[i];}
  std::vector<idx_t> & dim(idx_t i) {return impl.dim(i);}
  auto size() {return impl.size();}
  template <typename... Args>
  SymAccessList operator () (SymIterator symIterator, Args... args) 
  {
    std::vector<SymIterator> allIterators{{symIterator, args...}};

    SymAccess thisAccess = SymAccess(impl.get_data(), impl.get_permutation(), allIterators);
    thisAccess.isSparse = true;
    SymAccessList l = SymAccessList(thisAccess);
    return l ;
  }
  template <typename...Indices>
  ElmType & operator() (camp::idx_t i, Indices... is) {
    auto tuple = make_tuple(i,is...);
        return impl.operator()(tuple);
  }

   
  auto get_sparse_iteration_space(std::vector<camp::idx_t> dimOrder) {
    auto outerDim = new SparseSymbolicSegmentLead<NumDims>(*this, dimOrder[0]);
    auto innerDim = new SparseSymbolicSegmentFollow<NumDims>(*this, dimOrder[1], outerDim);
    return camp::tie(*outerDim, *innerDim);
  }

  size_t get_hits() {return impl.get_hits();}
  size_t get_misses() {return impl.get_misses();}
  void reset_counters() {impl.reset_counters();}
  double get_miss_rate() { return impl.get_miss_rate();}
  double get_hit_rate() { return impl.get_hit_rate();}
  void set_permutation(std::vector<idx_t> perm) {impl.set_permutation(perm);}

 
}; // SparseView<NumDims>


template <typename Tuple, camp::idx_t...Is>
auto tuple_to_vector(Tuple t, camp::idx_seq<Is...>) {
  auto vec = std::vector<std::vector<camp::idx_t>>{{camp::get<Is>(t)...}};
  return vec;
}

template <camp::idx_t...Is> 
auto seq_to_vector(camp::idx_seq<Is...>) {
  return std::vector<idx_t>{{Is...}};
}

template <typename ElmType, typename...Vectors>
auto make_sparse_view_permuted(Vectors...vecs) {
  auto tuple = camp::make_tuple(vecs...);
  size_t constexpr numDims = sizeof...(Vectors) - 2;

  auto val = camp::get<numDims>(tuple);
  auto permutation = camp::get<numDims+1>(tuple);
  auto indices = tuple_slice<0,numDims>(tuple);
  
  auto indicesVector = tuple_to_vector(indices, idx_seq_for(indices));

  COO<ElmType, numDims> impl(indicesVector, val, permutation);  

  return SparseView<ElmType, numDims, COO<ElmType, numDims>>(impl);

}

template <typename ElmType, typename...Vectors>
auto make_sparse_view_permuted_diag(Vectors...vecs) {
  auto tuple = camp::make_tuple(vecs...);
  size_t constexpr numDims = sizeof...(Vectors) - 2;

  auto val = camp::get<numDims>(tuple);
  auto permutation = camp::get<numDims+1>(tuple);
  auto indices = tuple_slice<0,numDims>(tuple);
  
  auto indicesVector = tuple_to_vector(indices, idx_seq_for(indices));

  DIAG<ElmType, numDims> impl(indicesVector, val, permutation);  

  return SparseView<ElmType, numDims, DIAG<ElmType, numDims>>(impl);

}

template <typename ElmType, typename...Vectors>
auto make_sparse_view(Vectors...vecs) {
  auto tuple = camp::make_tuple(vecs...);
  size_t constexpr numDims = sizeof...(Vectors) - 1;

  auto seq = idx_seq_from_to<0,numDims>();
  auto perm = seq_to_vector(seq);
  return make_sparse_view_permuted<ElmType>(vecs..., perm);
}

template <typename ElmType, typename...Vectors>
auto make_sparse_view_diag(Vectors...vecs) {
  auto tuple = camp::make_tuple(vecs...);
  size_t constexpr numDims = sizeof...(Vectors) - 1;

  auto seq = idx_seq_from_to<0,numDims>();
  auto perm = seq_to_vector(seq);
  return make_sparse_view_permuted_diag<ElmType>(vecs..., perm);
}


template <typename T>
RAJA::SparseView<T,2, COO<T,2>> make_random_sparse_view2(camp::idx_t Ni, camp::idx_t Nj, size_t perRowL, size_t perRowH ) {
  using namespace RAJA;
  std::vector<camp::idx_t> row;
  std::vector<camp::idx_t> col;

  //create a vector that we'll shuffle to use fill each row's entries
 
  auto fullCol = std::vector<camp::idx_t>(Nj);
  for(idx_t i = 0; i < Nj; i++) {fullCol[i] = i;}

  for(idx_t i = 0; i < Ni; i++) {
    //for each row, how many elements?
    auto numInRow = rand() % (perRowH-perRowL) + perRowL;
    auto thisRow = std::vector<camp::idx_t>(numInRow, i);
    
    std::random_shuffle(fullCol.begin(), fullCol.end());
    auto thisCol = std::vector<camp::idx_t>(fullCol.begin(), fullCol.begin() + numInRow);

    std::sort(thisCol.begin(), thisCol.end());

    row.insert(row.end(), thisRow.begin(), thisRow.end());
    col.insert(col.end(), thisCol.begin(), thisCol.end());

    //std::cout << "After inserting the " << i << "th row, we have:\n";
    for(size_t j = 0; j < row.size(); j++) {
    //  std::cout << "(" << row[j] << "," << col[j] << ")  ";
    }
    //std::cout << "\n";
  }
  
  std::vector<T> val(row.size());
  for(size_t i = 0; i < row.size(); i++) {
    T x = (T) rand() / (T) RAND_MAX;
    val[i] = x;
  }
  //std::cout << "creating sparse view. Here's whats passed in:\n";
  for(size_t i = 0; i < col.size(); i++) {
  //    std::cout << "(" << row[i] << "," << col[i] << ", " << val[i] << ")  ";
  }
  //std::cout << "\n";
  return make_sparse_view<T>(row,col,val);
}

template <typename T, bool diagonal=false, bool useDIAG=true>
auto make_random_sparse_view2(camp::idx_t N, double density = 0.1) {
  std::vector<idx_t> d0,d1;
  std::vector<T> v;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++) {
                
        bool isNonzero = (diagonal && i == j) || ((double) std::rand() / (double) RAND_MAX) < density;
        if(isNonzero) {
          d0.push_back(i);
          d1.push_back(j);
          T x = (T) std::rand() / (T) RAND_MAX;
          v.push_back(x);
        }
      
    }
  }

  if constexpr (useDIAG) {
    return make_sparse_view_diag<T>(d0,d1,v);
  } else {
    return make_sparse_view<T>(d0,d1,v);
  }
}


template <typename T>
RAJA::SparseView<T,3, COO<T,3>> make_random_sparse_view3(camp::idx_t N) {
  std::vector<idx_t> d0,d1,d2;
  std::vector<T> v;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++) {
      for(int k = 0; k < N; k++) {

        bool isNonzero = (std::rand() % 10) == 0;
        if(isNonzero) {
          d0.push_back(i);
          d1.push_back(j);
          d2.push_back(k);
          T x = (T) std::rand() / (T) RAND_MAX;
          v.push_back(x);
        }
      }
    }
  }

  return make_sparse_view<T>(d0,d1,d2,v);
}

template <typename T>
RAJA::SparseView<T,4, COO<T,4>> make_random_sparse_view4(camp::idx_t N) {
  std::vector<idx_t> d0,d1,d2,d3;
  std::vector<T> v;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++) {
      for(int k = 0; k < N; k++) {
      for(int l = 0; l < N; l++) {
        bool isNonzero = (std::rand() % 10) == 0;
        if(isNonzero) {
          d0.push_back(i);
          d1.push_back(j);
          d2.push_back(k);
          d3.push_back(l);
          T x = (T) std::rand() / (T) RAND_MAX;
          v.push_back(x);
        } 
        }
      }
    }
  }

  return make_sparse_view<T>(d0,d1,d2,d3,v);
}



} //namespace RAJA
#endif
