#pragma once
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <vector>

// Only included by SOP_PROFILE. GPU events add no extra synchronization:
// end-of-job collection queries events after the existing model completion waits.
namespace sop_profile {
using Clock = std::chrono::steady_clock;
inline double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
inline void check(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Row { const char* model; const char* stage; double duration; };
struct GpuMark { cudaEvent_t start{},end{}; const char* model{}; const char* stage{}; };
struct Output {
    std::mutex mutex;
    std::ofstream file;
    Output() {
        if(const char* path=std::getenv("SOP_PROFILE_CSV")) {
            file.open(path);
            if(!file) throw std::runtime_error("Cannot open SOP_PROFILE_CSV");
            file<<"lane,frame,start_ms,domain,model,stage,duration_ms\n"<<std::setprecision(9);
        }
    }
};
inline Output& output() { static Output value; return value; }
inline void standalone(int lane,uint64_t frame,double elapsed,const char* model,
                       const std::vector<Row>& rows) {
    auto& out=output(); if(!out.file.is_open()) return;
    std::lock_guard<std::mutex> lock(out.mutex);
    for(const auto& r:rows)
        out.file<<lane<<','<<frame<<','<<elapsed<<",cpu,"<<model<<','<<r.stage<<','<<r.duration<<'\n';
}
struct State {
    bool active=false;
    int lane=-1;
    uint64_t frame=0;
    double elapsed=0;
    const char* model="job";
    std::vector<Row> cpu;
    std::vector<GpuMark> gpu;
    size_t used=0;
    ~State() { for(auto& g:gpu) { cudaEventDestroy(g.start); cudaEventDestroy(g.end); } }
    void begin(int laneId,uint64_t frameId,double startMs) {
        active=output().file.is_open(); lane=laneId; frame=frameId; elapsed=startMs;
        model="job"; cpu.clear(); used=0;
    }
    void finish() {
        if(!active) return;
        auto& out=output(); std::lock_guard<std::mutex> lock(out.mutex);
        for(const auto& r:cpu)
            out.file<<lane<<','<<frame<<','<<elapsed<<",cpu,"<<r.model<<','<<r.stage<<','<<r.duration<<'\n';
        for(size_t i=0;i<used;++i) {
            const auto& g=gpu[i];
            // A not-ready event is an instrumentation error, never an implicit wait.
            check(cudaEventQuery(g.end));
            float duration=0; check(cudaEventElapsedTime(&duration,g.start,g.end));
            out.file<<lane<<','<<frame<<','<<elapsed<<",gpu,"<<g.model<<','<<g.stage<<','<<duration<<'\n';
        }
        active=false;
    }
};
inline State& state() { static thread_local State value; return value; }
struct CpuScope {
    State& state; const char* model; const char* stage; Clock::time_point start;
    explicit CpuScope(const char* name):state(sop_profile::state()),model(state.model),stage(name),start(Clock::now()) {}
    ~CpuScope() { if(state.active) state.cpu.push_back({model,stage,ms(start,Clock::now())}); }
};
struct GpuScope {
    State& state; cudaStream_t stream; size_t index=0; bool active;
    GpuScope(const char* name,cudaStream_t value):state(sop_profile::state()),stream(value),active(state.active) {
        if(!active) return;
        index=state.used++;
        if(index==state.gpu.size()) {
            GpuMark g; check(cudaEventCreate(&g.start)); check(cudaEventCreate(&g.end)); state.gpu.push_back(g);
        }
        auto& g=state.gpu[index]; g.model=state.model; g.stage=name; check(cudaEventRecord(g.start,stream));
    }
    ~GpuScope() { if(active) cudaEventRecord(state.gpu[index].end,stream); }
};
inline void extra(const char* model,const char* stage,double duration) {
    auto& s=state(); if(s.active) s.cpu.push_back({model,stage,duration});
}
}
