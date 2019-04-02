#pragma once

#include <vector>
#include <mutex>
#include <upcxx/upcxx.hpp>
#include "kmer_t.hpp"

struct HashMap {
  std::vector<upcxx::global_ptr<kmer_pair>> data;
  std::vector<upcxx::global_ptr<std::int32_t>> used;
  std::vector<kmer_pair> to_insert;
  // for hashtable miss tracking
  int num_uses;
  int tot_hits;
  size_t my_size;

  size_t size() const noexcept;

  HashMap(size_t size, std::vector<upcxx::global_ptr<kmer_pair>> _data, std::vector<upcxx::global_ptr<std::int32_t>> _used);

  // Most important functions: insert and retrieve
  // k-mers from the hash table.
  void insert(const kmer_pair &kmer, upcxx::atomic_domain<int>& ad);

  void queue(const kmer_pair &kmer);
  void reduce();
  bool find(const pkmer_t &key_kmer, kmer_pair &val_kmer);

  // Helper functions
    upcxx::global_ptr<kmer_pair> getSlotAddr(uint64_t slot);
    upcxx::global_ptr<std::int32_t> getUsedSlotAddr(uint64_t slot);
  // Write and read to a logical data slot in the table.
  void write_slot(uint64_t slot, const kmer_pair &kmer);
  kmer_pair read_slot(uint64_t slot);

  // Request a slot or check if it's already used.
  bool request_slot(uint64_t slot, upcxx::atomic_domain<int>& ad);
  bool slot_used(uint64_t slot);
};

HashMap::HashMap(size_t size, std::vector<upcxx::global_ptr<kmer_pair>> _data, std::vector<upcxx::global_ptr<std::int32_t>> _used) {
  my_size = size;
  data = _data;
  used = _used;
  num_uses = 0;
  tot_hits = 0;
}

// only insert "locally"
void HashMap::insert(const kmer_pair &kmer, upcxx::atomic_domain<int>& ad) {
  uint64_t hash = kmer.hash();
  uint64_t probe = 0;
  bool success = false;

  do {
    uint64_t slot = (hash + probe++) % size();
    success = request_slot(slot, ad);
    if (success) {
      write_slot(slot, kmer);
    }
  } while (!success && probe < size());
}

upcxx::global_ptr<kmer_pair> HashMap::getSlotAddr(uint64_t slot) {
    int node_num = slot / (size() / upcxx::rank_n());
    int offset = slot % (size() / upcxx::rank_n());
    return data[node_num] + offset;
}

upcxx::global_ptr<std::int32_t> HashMap::getUsedSlotAddr(uint64_t slot) {
    int node_num = slot / (size() / upcxx::rank_n());
    int offset = slot % (size() / upcxx::rank_n());
    return used[node_num] + offset;
}

bool HashMap::find(const pkmer_t &key_kmer, kmer_pair &val_kmer) {
    num_uses++;
  uint64_t hash = key_kmer.hash();
  uint64_t probe = 0;
  bool success = false;
  do {
    uint64_t slot = (hash + probe++) % size();
    //if (slot_used(slot, node_num)) {
      val_kmer = read_slot(slot);
      if (val_kmer.kmer == key_kmer) {
        success = true;
      }
      tot_hits++;
    //}
  } while (!success && probe < size());
  return success;
}

bool HashMap::slot_used(uint64_t slot) {
    // this function is currently not used
    return false;
    //return used[slot] != 0;
    /*
    upcxx::future<int> res = upcxx::rget(used[node_num] + slot);
    res.wait();
    return res.result() != 0;*/
  // return used[slot] != 0;
}

void HashMap::write_slot(uint64_t slot, const kmer_pair &kmer) {
    upcxx::rput(kmer, getSlotAddr(slot)).wait();
}


kmer_pair HashMap::read_slot(uint64_t slot) {
    upcxx::future<kmer_pair> res = upcxx::rget(getSlotAddr(slot));
    res.wait();
    return res.result();
  //return data[slot];
}

bool HashMap::request_slot(uint64_t slot, upcxx::atomic_domain<int>& ad) {
    upcxx::future<std::int32_t> res =  ad.fetch_add(getUsedSlotAddr(slot), 1, std::memory_order_relaxed);
    res.wait();
  if (res.result() != 0) {
    return false;
  } else {
    return true;
  }
}

size_t HashMap::size() const noexcept {
  return my_size;
}
