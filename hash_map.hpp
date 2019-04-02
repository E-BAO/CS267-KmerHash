#pragma once

#include <vector>
#include <mutex>
#include <upcxx/upcxx.hpp>
#include "kmer_t.hpp"

struct HashMap {
  std::vector<upcxx::global_ptr<kmer_pair>> data;
  std::vector<upcxx::global_ptr<std::int32_t>> used;
  std::vector<kmer_pair> to_insert;

  size_t my_size;
  size_t total_size;
  size_t nprocs;

  size_t size() const noexcept;

  HashMap(size_t size);

  // Most important functions: insert and retrieve
  // k-mers from the hash table.
  bool insert(const kmer_pair &kmer, upcxx::atomic_domain<int>& ad);
  bool find(const pkmer_t &key_kmer, kmer_pair &val_kmer);

  // Helper functions to compute global index
  upcxx::global_ptr<kmer_pair> getSlotAddr(uint64_t slot);
  upcxx::global_ptr<std::int32_t> getUsedSlotAddr(uint64_t slot);

  // Write and read to a logical data slot in the table.
  void write_slot(uint64_t slot, const kmer_pair &kmer);
  kmer_pair read_slot(uint64_t slot);

  // Request a slot or check if it's already used.
  bool request_slot(uint64_t slot, upcxx::atomic_domain<int>& ad);
  bool slot_used(uint64_t slot);
};

HashMap::HashMap(size_t size){
  nprocs = upcxx::rank_n();
  total_size = size;
  my_size = (size + nprocs - 1) / nprocs;

  data.resize(nprocs);
  used.resize(nprocs, 0);

  for (int i = 0; i < nprocs; i++) {
    if(upcxx::rank_me() == i){
      data[i] = upcxx::new_array<kmer_pair>(my_size);
      used[i] = upcxx::new_array<int>(my_size);
    }
    data[i] = upcxx::broadcast(data[i], i).wait();
    used[i] = upcxx::broadcast(used[i], i).wait();
  }
}

bool HashMap::insert(const kmer_pair &kmer, upcxx::atomic_domain<int>& ad) {
  uint64_t hash = kmer.hash();
  uint64_t probe = 0;
  bool success = false;
  do {
    uint64_t slot = (hash + probe++) % total_size;
    success = request_slot(slot, ad);
    if (success) {
      write_slot(slot, kmer);
    }
  } while (!success && probe < total_size);
  return success;
}

upcxx::global_ptr<kmer_pair> HashMap::getSlotAddr(uint64_t slot) {
    int node_num = slot / (total_size / nprocs);
    int offset = slot % (total_size / nprocs);
    return data[node_num] + offset;
}

upcxx::global_ptr<std::int32_t> HashMap::getUsedSlotAddr(uint64_t slot) {
    int node_num = slot / (total_size / nprocs);
    int offset = slot % (total_size / nprocs);
    return used[node_num] + offset;
}

bool HashMap::find(const pkmer_t &key_kmer, kmer_pair &val_kmer) {
  uint64_t hash = key_kmer.hash();
  uint64_t probe = 0;
  bool success = false;
  do {
    uint64_t slot = (hash + probe++) % total_size;
    if (slot_used(slot)) {
      val_kmer = read_slot(slot);
      if (val_kmer.kmer == key_kmer) {
        success = true;
      }
    }
  } while (!success && probe < total_size);
  return success;
}

bool HashMap::slot_used(uint64_t slot) {
  // asynchronous, get value from remote rank
    upcxx::future<int> res = upcxx::rget(getUsedSlotAddr(slot));
    res.wait();
    return res.result() != 0;
}

void HashMap::write_slot(uint64_t slot, const kmer_pair &kmer) {
    upcxx::rput(kmer, getSlotAddr(slot)).wait();
}


kmer_pair HashMap::read_slot(uint64_t slot) {
    return upcxx::rget(getSlotAddr(slot)).wait();
}

bool HashMap::request_slot(uint64_t slot, upcxx::atomic_domain<int>& ad) {
  upcxx::future<std::int32_t> res =  ad.fetch_add(getUsedSlotAddr(slot), 1, std::memory_order_relaxed);
  res.wait();
  return res.result() == 0;
}

size_t HashMap::size() const noexcept {
  return my_size;
}
