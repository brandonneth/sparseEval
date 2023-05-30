#ifndef _RAJA_COO_HPP
#define _RAJA_COO_HPP


#include "RAJA/loopchain/Sparse/SparseImpl.hpp"

namespace RAJA {
template <typename ElmType, size_t NumDims>
struct COO : public SparseImpl<ElmType, NumDims> {

  using IdxType = camp::idx_t;
  using VectorType = std::vector<IdxType>;
  std::vector<VectorType> indices;
 

  std::vector<ElmType> val;
  std::vector<IdxType> permutation;
  ElmType fill = 0;
  const size_t len;

  COO(std::vector<VectorType> t, std::vector<ElmType> v, std::vector<IdxType> p) 
    : indices(t), val(v), permutation(p), len(val.size()) {
    verify();
  }

  void verify() {
    //check lengths
    //check that its sorted the right way
    //check the permutation is right
    for(size_t i = 0; i < NumDims; i++) {
      VectorType idxs = indices.at(i);
      assert(val.size() == idxs.size());
    }
    sort();
  }

  std::vector<IdxType> dim(IdxType t) {
    return indices[t];
  }
  template <IdxType...Is>
  auto make_entry(IdxType n, camp::idx_seq<Is...>) {
    std::vector<IdxType> i{indices[Is][n]...};
    return camp::make_tuple(i, val[n]);
  }

  void sort() {
    auto valTuple = camp::make_tuple(val[0]);
    auto indexTuple = tuple_repeat<NumDims>(indices[0][0]);
    auto entryTuple = tuple_cat(indexTuple, valTuple);
    camp::sink(entryTuple); // quiets use-def warning 
    using EntryType = camp::tuple<std::vector<IdxType>, ElmType>;
    std::vector<EntryType> entries;

    for(size_t i = 0 ; i < len; i++) {
      EntryType next = make_entry(i, idx_seq_from_to<0,NumDims>());

      entries.push_back(next);
    }
 
    auto comp = [=](auto a, auto b) {

      auto aa = camp::get<0>(a);
      auto bb = camp::get<0>(b);
      for(size_t i = 0; i < NumDims; i++) {
        if (aa[permutation[i]] != bb[permutation[i]]) {
          return aa[permutation[i]] < bb[permutation[i]];
        }
      }
      return false;
    } ;

    std::sort(entries.begin(), entries.end(), comp);
    
    for(size_t i = 0; i < len; i++) {
      auto newIndices = camp::get<0>(entries[i]);
      auto newValue = camp::get<1>(entries[i]);
      for(size_t j = 0; j < NumDims; j++) {
        indices[j][i] = newIndices[j];
      }
      val[i] = newValue;
    }
  
  }

  idx_t expectedIdx = 0; 
  ElmType & operator() (std::vector<IdxType> v) { 
    if(expectedIdx >= len) {expectedIdx = 0;}
    int matches = 1;
    for(size_t i = 0; i < NumDims; i++) {
      if (v[i] != indices[i][expectedIdx]) {
        matches = 0;
        break;
      }
    }
    if (matches) {
      hits += 1;
      
      return val[expectedIdx++];
    } else {
      misses += 1;
      auto entryIndex = find<0>(0,len,v);
      if (entryIndex == -1) {
        return fill;
      } else {
        expectedIdx = entryIndex + 1;
        return val[entryIndex];
      }
    }
  }
  
  void set_permutation(std::vector<IdxType> p) {
    permutation = p;
    sort();
  } 
 
  template<size_t CurrDepth>
  idx_t find(size_t begin, size_t end, std::vector<IdxType> v) { 
    auto currCoord = indices[permutation[CurrDepth]];
    auto currBegin = currCoord.begin() + begin;
    auto currEnd = currCoord.begin() + end;
    
    
    auto currFind = std::find(currBegin, currEnd, v[permutation[CurrDepth]]);
    if (currFind == currEnd) {return -1;}
   

    auto currFindEnd = std::find_end(currFind, currEnd, currFind, currFind + 1) + 1;
    
    auto nextBeginDist = begin + std::distance(currBegin, currFind);
    auto nextEndDist = begin + std::distance(currBegin, currFindEnd);

    if constexpr (CurrDepth == NumDims - 1) {
      camp::sink(nextEndDist); // silence unused by set warning
      return nextBeginDist;
    } else {
      return find<CurrDepth+1>(nextBeginDist, nextEndDist, v);
    }
  }

  std::vector<IdxType> get_permutation(){ return permutation;};  
  void* get_data(){ return val.data();};  
 
  size_t size() { return val.size();}
 
  size_t hits = 0;
  size_t misses = 0;
  virtual size_t get_hits() {return hits;}
  virtual size_t get_misses() {return misses;}
  virtual void reset_counters() {
    hits = 0;
    misses = 0;
  }

}; //COO


} //namespace RAJA

#endif
