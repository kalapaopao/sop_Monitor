// Included after TensorResult, inside the existing anonymous namespace.
struct SopPinned {
    void* data=nullptr; size_t size=0; uint64_t allocations=0;
    SopPinned()=default;
    SopPinned(const SopPinned&)=delete;
    SopPinned& operator=(const SopPinned&)=delete;
    ~SopPinned() { if(data) cudaFreeHost(data); }
    void* ensure(size_t bytes) {
        if(bytes>size) {
            if(data) checkCuda(cudaFreeHost(data),"free pinned storage");
            data=nullptr; size=0;
            checkCuda(cudaHostAlloc(&data,bytes,cudaHostAllocDefault),"allocate pinned storage");
            size=bytes; ++allocations;
        }
        return data;
    }
};
struct SopStage2IO {
    SopPinned source, input;
    std::array<SopPinned,16> output;
    std::vector<TensorResult> results;
    cudaEvent_t completed=nullptr;
    void* nativeDevice=nullptr; size_t nativeBytes=0;
    bool prepared=false;
    uint64_t resultAllocations=0, uploadBytes=0;
    ~SopStage2IO() {
        if(completed) cudaEventDestroy(completed);
        if(nativeDevice) cudaFree(nativeDevice);
    }
};
