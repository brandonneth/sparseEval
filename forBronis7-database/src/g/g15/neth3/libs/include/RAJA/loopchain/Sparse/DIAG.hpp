#ifndef _RAJA_DIAG_HPP
#define _RAJA_DIAG_HPP


#include "RAJA/loopchain/Sparse/SparseImpl.hpp"

namespace RAJA {
template <typename ElmType, size_t NumDims>
struct DIAG : public SparseImpl<ElmType, NumDims,DIAG<ElmType,NumDims>> {

  using IdxType = camp::idx_t;
  using VectorType = std::vector<IdxType>;
  std::vector<VectorType> indices;
 

  std::vector<ElmType> val;
  std::vector<IdxType> permutation;
  std::vector<ElmType> diag;
  ElmType fill = 0;
  const size_t len;

  DIAG(std::vector<VectorType> t, std::vector<ElmType> v, std::vector<IdxType> p) 
    : indices(t), val(v), permutation(p), len(val.size()) {
    verify();
    auto count = dim(0).back();

    diag = std::vector<ElmType>(count+1);
    for(size_t i = 0; i < len; i++) {
      size_t idx = dim(0)[i];
      if(is_diagonal_entry(i, idx)) {
        diag.at(idx) = val[i];
      }
    } 
  }

  template <camp::idx_t Idx=1>
  RAJA_INLINE
  bool is_diagonal_entry(const idx_t & entryIdx, const idx_t & idxVal) {
    if constexpr (Idx == NumDims) { 
      return true; 
    } else {
      return idxVal == indices[Idx][entryIdx] && is_diagonal_entry<Idx+1>(entryIdx, idxVal);
    }

  }

  RAJA_INLINE
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

  RAJA_INLINE
  std::vector<IdxType> & dim(IdxType t) {
    return indices[t];
  }
  template <IdxType...Is>
  RAJA_INLINE
  auto make_entry(IdxType n, camp::idx_seq<Is...>) {
    std::array<IdxType,NumDims> i{indices[Is][n]...};
    return camp::make_tuple(i, val[n]);
  }

  RAJA_INLINE
  void sort() {
    auto valTuple = camp::make_tuple(val[0]);
    auto indexTuple = tuple_repeat<NumDims>(indices[0][0]);
    auto entryTuple = tuple_cat(indexTuple, valTuple);
    camp::sink(entryTuple); // quiets use-def warning 
    using EntryType = camp::tuple<std::array<IdxType,NumDims>, ElmType>;
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

  template <idx_t I, typename...Idxs>
  RAJA_INLINE
  bool is_expected(camp::tuple<Idxs...> idxs) const {
    if constexpr (sizeof...(Idxs) == I) {
      return 1;
    } else {
      auto thisMatch = (camp::get<I>(idxs) == indices[I][expectedIdx]);
      return thisMatch && is_expected<I+1>(idxs);
    }
  }

  template <idx_t I, typename...Idxs>
  RAJA_INLINE
  bool is_post_diagonal(camp::tuple<Idxs...> idxs) const {
    if constexpr (sizeof...(Idxs) == I) {
      return 1;
    } else {
      auto thisMatch = (camp::get<I>(idxs) == indices[I][expectedIdx+1]);
      return thisMatch && is_post_diagonal<I+1>(idxs);
    }
  }

  template <camp::idx_t RecIdx, typename...Idxs>
  RAJA_INLINE
  bool is_diagonal(camp::tuple<Idxs...> idxs) const {
    if constexpr (RecIdx == sizeof...(Idxs)) {
      return 1;
    } else {
      return camp::get<0>(idxs) == camp::get<RecIdx>(idxs) && is_diagonal<RecIdx+1>(idxs);
    }
  }


  template <typename...Idxs, camp::idx_t...Is>
  RAJA_INLINE
  std::array<idx_t, NumDims> tuple_to_array(camp::tuple<Idxs...> idxs, camp::idx_seq<Is...>) {
    return std::array<idx_t, NumDims>{{camp::get<Is>(idxs)...}};
  }

  template <typename...Idxs>
  RAJA_INLINE
  std::array<idx_t, NumDims> tuple_to_array(camp::tuple<Idxs...> idxs) {
    return tuple_to_array(idxs, idx_seq_for(idxs));
  }

  template <typename...Idxs>
  RAJA_INLINE
  ElmType & operator() (camp::tuple<Idxs...> idxs) {
    
    if(expectedIdx >= len) {expectedIdx = 0;}
    if(is_expected<0>(idxs)) {
      hits += 1;
      return val[expectedIdx++];
    } else if(is_diagonal<0>(idxs)) {
      hits += 1;
      return diag[camp::get<0>(idxs)];
    } else if (is_post_diagonal<0>(idxs)) {
      hits += 1;
      expectedIdx++;
      return val[expectedIdx++];
    } else {
      misses += 1;
      
      auto entryIndex = find<0>(0,len,tuple_to_array(idxs));
      if (entryIndex == -1) {
        return fill;
      } else {
        expectedIdx = entryIndex + 1;
        return val[entryIndex];
      }
    }


  }


  template <idx_t I>
  RAJA_INLINE bool is_expected(const std::array<idx_t, NumDims> & v) {
    if constexpr (I == NumDims) {
      return true;
    } else {
    return v[I] == indices[I][expectedIdx] && is_expected<I+1>(v);
    }
  }


  
  ElmType & operator() (const std::array<idx_t, NumDims> & v) { 
    if((size_t)expectedIdx >= len) {expectedIdx = 0;}
    int matches = 1;
//is_expected<0>(v);
    for(size_t i = 0; i < NumDims; i++) {
      if (v[i] != indices[i][expectedIdx]) { matches = 0; break;}
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
 

  template<size_t CurrDepth, typename T>
  RAJA_INLINE
  idx_t find(size_t begin, size_t end, T v) { 
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
