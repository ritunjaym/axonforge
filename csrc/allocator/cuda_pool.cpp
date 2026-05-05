/*
 * AxonForge CUDA Caching Allocator
 *
 * A size-bucketed free-list caching allocator on top of cudaMalloc.
 * Plugs into PyTorch 2.x via torch.cuda.memory.CUDAPluggableAllocator.
 *
 * Design:
 *   Size classes: powers of 2 from 256B to 128MB.
 *     <128KB  → small pool (high-frequency small allocs)
 *     <128MB  → large pool (parameter/gradient-sized allocs)
 *     ≥256MB  → passthrough (direct cudaMalloc/cudaFree, never pooled)
 *
 *   Allocation (allocate):
 *     1. Round size up to next power-of-2 size class.
 *     2. Lock mutex; search free_lists_[size_class].
 *     3. Cache hit  → pop Block, mark in_use, return ptr. No cudaMalloc call.
 *     4. Cache miss → cudaMalloc(size_class), create Block*, insert into block_map_.
 *     5. Update peak_allocated_bytes_, num_cudamalloc_calls_, total_allocs_.
 *
 *   Deallocation (deallocate):
 *     1. Lock mutex; look up Block* in block_map_[ptr].
 *     2. Coalesce: check prev_mem/next_mem neighbours; if free, merge.
 *     3. If merged size ≥ kPassthroughThreshold: cudaFree; erase from block_map_.
 *     4. Else: mark in_use=false; push onto free_lists_[rounded_size].
 *
 *   Fragmentation:
 *     frag_pct = (total_cuda_bytes - in_use_bytes) / total_cuda_bytes * 100
 *     Logged every 100 allocations; indicates how much pooled memory is idle.
 *
 *   Thread safety: std::mutex guards all free_lists_ and block_map_ access.
 *
 * PyTorch integration (Python side):
 *   import axonforge_allocator
 *   axonforge_allocator.enable()    ← registers via CUDAPluggableAllocator
 *   axonforge_allocator.disable()   ← resets (note: PyTorch 2.x is one-way)
 *   axonforge_allocator.stats()     ← returns metrics dict
 *   axonforge_allocator.reset_stats() ← zeroes counters between tests
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <atomic>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            abort();                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Size class constants
// ---------------------------------------------------------------------------

static constexpr size_t kMinSizeClass      = 256;
static constexpr size_t kSmallThreshold    = 128ULL * 1024;           // 128 KB
static constexpr size_t kLargeThreshold    = 128ULL * 1024 * 1024;   // 128 MB
static constexpr size_t kPassthroughThresh = 256ULL * 1024 * 1024;   // 256 MB

static size_t round_up_size_class(size_t n) {
    if (n == 0) return kMinSizeClass;
    size_t sc = kMinSizeClass;
    while (sc < n) sc <<= 1;
    return sc;
}

// ---------------------------------------------------------------------------
// Block — one allocation unit, kept in memory-address order for coalescing
// ---------------------------------------------------------------------------

struct Block {
    void*   ptr          = nullptr;
    size_t  size         = 0;       // size class (rounded-up)
    bool    in_use       = false;
    Block*  prev_mem     = nullptr; // lower-address neighbour
    Block*  next_mem     = nullptr; // higher-address neighbour
};

// ---------------------------------------------------------------------------
// CachingAllocator
// ---------------------------------------------------------------------------

class CachingAllocator {
public:
    // Allocate from pool or cudaMalloc
    void* allocate(size_t size, int device, cudaStream_t /*stream*/) {
        if (size == 0) return nullptr;

        // Large blocks bypass the pool entirely
        if (size >= kPassthroughThresh) {
            void* ptr = nullptr;
            {
                c10::cuda::CUDAGuard guard(device);
                CUDA_CHECK(cudaMalloc(&ptr, size));
            }
            std::lock_guard<std::mutex> lock(mutex_);
            ++num_cudamalloc_calls_;
            ++total_allocs_;
            large_passthrough_set_.insert(ptr);
            return ptr;
        }

        size_t sc = round_up_size_class(size);

        std::lock_guard<std::mutex> lock(mutex_);
        ++total_allocs_;

        auto& free_list = free_lists_[sc];
        if (!free_list.empty()) {
            Block* b = free_list.back();
            free_list.pop_back();
            b->in_use = true;
            in_use_bytes_ += b->size;
            ++num_cache_hits_;
            maybe_log_fragmentation();
            return b->ptr;
        }

        // Cache miss — allocate from CUDA
        void* ptr = nullptr;
        {
            c10::cuda::CUDAGuard guard(device);
            CUDA_CHECK(cudaMalloc(&ptr, sc));
        }
        ++num_cudamalloc_calls_;
        total_cuda_bytes_ += sc;
        peak_allocated_bytes_ = std::max(peak_allocated_bytes_, total_cuda_bytes_);

        Block* b    = new Block{ptr, sc, true, nullptr, nullptr};
        block_map_[ptr] = b;
        in_use_bytes_ += sc;

        maybe_log_fragmentation();
        return ptr;
    }

    void deallocate(void* ptr, size_t /*size*/, int /*device*/, cudaStream_t /*stream*/) {
        if (!ptr) return;

        std::lock_guard<std::mutex> lock(mutex_);

        // Large passthrough — return directly to CUDA
        if (large_passthrough_set_.count(ptr)) {
            large_passthrough_set_.erase(ptr);
            CUDA_CHECK(cudaFree(ptr));
            return;
        }

        auto it = block_map_.find(ptr);
        if (it == block_map_.end()) {
            // Unknown pointer — forward to cudaFree (safety fallback)
            CUDA_CHECK(cudaFree(ptr));
            return;
        }

        Block* b = it->second;
        b->in_use = false;
        in_use_bytes_ -= b->size;

        // Coalesce with next neighbour
        if (b->next_mem && !b->next_mem->in_use) {
            Block* nxt = b->next_mem;
            // Remove nxt from its free list
            remove_from_free_list(nxt);
            // Merge: b absorbs nxt
            total_cuda_bytes_ -= nxt->size;  // we'll add b's new size below
            b->size += nxt->size;
            total_cuda_bytes_ += b->size;    // (net: nxt->size added to b)
            b->next_mem = nxt->next_mem;
            if (nxt->next_mem) nxt->next_mem->prev_mem = b;
            block_map_.erase(nxt->ptr);
            delete nxt;
            // Re-key b in block_map_ (ptr didn't change)
        }

        // Coalesce with prev neighbour
        if (b->prev_mem && !b->prev_mem->in_use) {
            Block* prv = b->prev_mem;
            remove_from_free_list(prv);
            total_cuda_bytes_ -= b->size;
            prv->size += b->size;
            total_cuda_bytes_ += prv->size;
            prv->next_mem = b->next_mem;
            if (b->next_mem) b->next_mem->prev_mem = prv;
            block_map_.erase(b->ptr);
            delete b;
            b = prv;
        }

        // If coalesced block is above threshold, return to CUDA
        if (b->size >= kPassthroughThresh) {
            total_cuda_bytes_ -= b->size;
            block_map_.erase(b->ptr);
            CUDA_CHECK(cudaFree(b->ptr));
            delete b;
            return;
        }

        // Otherwise push onto free list for its (now possibly larger) size class
        free_lists_[b->size].push_back(b);
        maybe_log_fragmentation();
    }

    // Stats
    struct Stats {
        size_t peak_bytes;
        size_t total_allocs;
        size_t num_cache_hits;
        size_t num_cudamalloc_calls;
        double fragmentation_pct;
    };

    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        double frag = 0.0;
        if (total_cuda_bytes_ > 0) {
            frag = (double)(total_cuda_bytes_ - in_use_bytes_)
                 / (double)total_cuda_bytes_ * 100.0;
        }
        return {
            peak_allocated_bytes_,
            total_allocs_,
            num_cache_hits_,
            num_cudamalloc_calls_,
            frag,
        };
    }

    void reset_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        peak_allocated_bytes_ = 0;
        total_allocs_         = 0;
        num_cache_hits_       = 0;
        num_cudamalloc_calls_ = 0;
        alloc_since_log_      = 0;
    }

private:
    mutable std::mutex mutex_;

    // free_lists_[size_class] = vector of free Blocks of that size
    std::unordered_map<size_t, std::vector<Block*>> free_lists_;
    // block_map_[ptr] = Block* for every pooled allocation
    std::unordered_map<void*, Block*>               block_map_;
    // Large passthrough allocations (not pooled)
    std::unordered_set<void*>                       large_passthrough_set_;

    size_t peak_allocated_bytes_  = 0;
    size_t total_allocs_          = 0;
    size_t num_cache_hits_        = 0;
    size_t num_cudamalloc_calls_  = 0;
    size_t total_cuda_bytes_      = 0;  // bytes currently held from CUDA
    size_t in_use_bytes_          = 0;  // bytes currently handed to user
    size_t alloc_since_log_       = 0;

    void remove_from_free_list(Block* b) {
        auto& fl = free_lists_[b->size];
        fl.erase(std::remove(fl.begin(), fl.end(), b), fl.end());
    }

    void maybe_log_fragmentation() {
        if (++alloc_since_log_ >= 100) {
            alloc_since_log_ = 0;
            if (total_cuda_bytes_ > 0) {
                double frag = (double)(total_cuda_bytes_ - in_use_bytes_)
                            / (double)total_cuda_bytes_ * 100.0;
                if (frag > 30.0) {
                    fprintf(stderr, "[axonforge_allocator] fragmentation=%.1f%% "
                            "(cuda=%zu MB, in_use=%zu MB)\n",
                            frag,
                            total_cuda_bytes_ / (1024 * 1024),
                            in_use_bytes_    / (1024 * 1024));
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static CachingAllocator& global_allocator() {
    static CachingAllocator inst;
    return inst;
}

// ---------------------------------------------------------------------------
// C-linkage functions for CUDAPluggableAllocator
// ---------------------------------------------------------------------------

extern "C" {

void* axonforge_alloc(size_t size, int device, cudaStream_t stream) {
    return global_allocator().allocate(size, device, stream);
}

void axonforge_free(void* ptr, size_t size, int device, cudaStream_t stream) {
    global_allocator().deallocate(ptr, size, device, stream);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Python binding
// ---------------------------------------------------------------------------

PYBIND11_MODULE(axonforge_allocator, m) {
    m.doc() = "AxonForge CUDA Caching Allocator";

    m.def("enable", []() {
        // Register our allocator with PyTorch via CUDAPluggableAllocator
        // The .so path is found via importlib at Python call time.
        py::module_ torch_mem = py::module_::import("torch").attr("cuda").attr("memory");
        py::module_ importlib  = py::module_::import("importlib.util");
        py::object  spec       = importlib.attr("find_spec")("axonforge_allocator");
        std::string lib_path   = spec.attr("origin").cast<std::string>();

        py::object pluggable = torch_mem.attr("CUDAPluggableAllocator")(
            lib_path, "axonforge_alloc", "axonforge_free"
        );
        torch_mem.attr("change_current_allocator")(pluggable);
    });

    m.def("disable", []() {
        // PyTorch 2.x does not support reverting the allocator within a process.
        // In tests, call this for logical clarity; restart process to truly reset.
        fprintf(stderr, "[axonforge_allocator] disable() called — "
                "allocator change is permanent for this process lifetime.\n");
    });

    m.def("reset_stats", []() {
        global_allocator().reset_stats();
    });

    m.def("stats", []() -> py::dict {
        auto s = global_allocator().get_stats();
        double hit_rate = s.total_allocs > 0
            ? (double)s.num_cache_hits / (double)s.total_allocs * 100.0
            : 0.0;
        return py::dict(
            py::arg("peak_allocated_mb")    = (double)s.peak_bytes / 1e6,
            py::arg("num_cudaMalloc_calls") = (py::int_)s.num_cudamalloc_calls,
            py::arg("cache_hit_rate_pct")   = hit_rate,
            py::arg("fragmentation_pct")    = s.fragmentation_pct
        );
    });
}
