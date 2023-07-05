#ifndef _RAJA_SparseSegment_HPP
#define _RAJA_SparseSegment_HPP

#include "RAJA/loopchain/SymbolicSegment/SymbolicSegment.hpp"

namespace RAJA {


struct SparseIteratorImpl {
  virtual idx_t operator[](size_t i) = 0;
  virtual idx_t index() = 0;
  virtual idx_t update(idx_t i) = 0;
};



struct SparseIterator {
  SparseIteratorImpl * impl;
  SparseIterator (SparseIteratorImpl * _impl) : impl(_impl) {}
  idx_t operator[](size_t i ) {
    return impl->operator[](i);
  }
  idx_t operator - (SparseIterator other) {
    return impl->index() - other.impl->index();
  }
  idx_t update(idx_t i) {
    return impl->update(i);
  }
};

struct SparseSegmentImpl {
  virtual SparseIterator begin() = 0;
  virtual SparseIterator end() = 0;
} ;

struct SparseSegment {
  using iterator = SparseIterator;
  using IndexType = idx_t;

  SparseSegmentImpl * impl;
  SparseSegment(SparseSegmentImpl * impl) :impl(impl) {}
  SparseIterator begin() const {return impl->begin();}
  SparseIterator end() const {return impl->end();}
};

class CompressedFollowIterator : public SparseIteratorImpl {
  std::vector<idx_t> dim;
  idx_t offset;
public:
  idx_t rowStart;
  
  CompressedFollowIterator(std::vector<idx_t> dim, idx_t offset) 
    : dim(dim), offset(offset) {
    rowStart = 0;
  }

  idx_t operator[](size_t i) {
    return dim[rowStart + offset + i];
  }
  idx_t index() {
    return rowStart;
  }
  idx_t update(idx_t i) {
    return i;
  }
};

class CompressedFollowSegment : public SparseSegmentImpl {
  std::vector<idx_t> dim;
  CompressedFollowIterator *beginImpl;
  CompressedFollowIterator *endImpl;
  using iterator = SparseIterator;
  using IndexType = idx_t;
public:
  CompressedFollowSegment(std::vector<idx_t> dim) 
    : dim(dim) {
    beginImpl = new CompressedFollowIterator(dim,0);
    endImpl = new CompressedFollowIterator(dim,dim.size());
  }
  SparseIterator begin() {return SparseIterator(beginImpl);}
  SparseIterator end() {return SparseIterator(endImpl);}
  void update(idx_t iLow, idx_t iHigh) {
    beginImpl->rowStart = iLow;
    endImpl->rowStart = iHigh;
  }
};

class CompressedLeadIterator : public SparseIteratorImpl {
public:
  std::vector<idx_t> dimStarts;
  idx_t offset;
  std::vector<CompressedFollowSegment *> followers;
  CompressedLeadIterator(std::vector<idx_t> dimStarts, idx_t offset, std::vector<CompressedFollowSegment *> followers)
    : dimStarts(dimStarts), offset(offset), followers(followers) {}

  idx_t operator[](size_t i) {return i;}
  idx_t index() {
    return offset;
  }
  idx_t update(idx_t i) {
    for(size_t idx = 0; idx < followers.size(); idx++) {
      followers[idx]->update(dimStarts[i], dimStarts[i+1]);
    }
    return i;
  }
};

class CompressedLeadSegment : public SparseSegmentImpl {
  std::vector<idx_t> dimStarts;
  CompressedLeadIterator *beginImpl;
  CompressedLeadIterator *endImpl;
  std::vector<CompressedFollowSegment *> followers;
  using iterator = SparseIterator;
  using IndexType = idx_t;
public:
  CompressedLeadSegment(std::vector<idx_t> dimStarts, std::vector<CompressedFollowSegment *> followers)
    : dimStarts(dimStarts), followers(followers) {
 
    beginImpl = new CompressedLeadIterator(dimStarts, 0, followers);
    endImpl = new CompressedLeadIterator(dimStarts, dimStarts.size() - 1, followers); //minus one bc of the +1 indexing
  }
  //TODO: destructor to free impls
  
  SparseIterator begin() {return SparseIterator(beginImpl);}
  SparseIterator end() {return SparseIterator(endImpl);};

};



template <typename...FollowSegs, idx_t...Idxs>
auto make_compressed_lead_segment(std::vector<idx_t> dim, camp::tuple<FollowSegs...> followSegs, camp::idx_seq<Idxs...>) {
  std::vector<idx_t> indices;
  for(size_t dimIndex = 0; dimIndex < dim.size(); dimIndex++) {
    auto dimValue = dim[dimIndex];
    while ((idx_t) indices.size() <= dimValue) {
      indices.push_back(dimIndex);
    }
  }
  indices.push_back(dim.size());

  std::vector<CompressedFollowSegment*> followers = { (CompressedFollowSegment*) camp::get<Idxs>(followSegs).impl ...};
  auto leadSeg = new CompressedLeadSegment(indices, followers);
  return SparseSegment(leadSeg);
}

template <typename...FollowSegs>
auto make_compressed_lead_segment(std::vector<idx_t> dim, camp::tuple<FollowSegs...> followSegs) {
  return make_compressed_lead_segment(dim,followSegs, idx_seq_for(followSegs));
}


auto make_compressed_lead_segment(std::vector<idx_t> dim, SparseSegment followSeg) {
  return make_compressed_lead_segment(dim,make_tuple(followSeg));
}



auto make_compressed_follow_segment(std::vector<idx_t> dim) {
  auto followSeg = new CompressedFollowSegment(dim);
  return SparseSegment(followSeg);
}

} //namespace RAJA

template <>
struct std::iterator_traits<RAJA::SparseIterator> {
  typedef camp::idx_t value_type;
  typedef camp::idx_t difference_type;
  typedef camp::idx_t* pointer;
  typedef camp::idx_t& reference;
  typedef std::random_access_iterator_tag iterator_category;
};


template <>
struct std::iterator_traits<RAJA::SparseIteratorImpl> {
  typedef camp::idx_t value_type;
  typedef camp::idx_t difference_type;
  typedef camp::idx_t* pointer;
  typedef camp::idx_t& reference;
  typedef std::random_access_iterator_tag iterator_category;
};


#endif
