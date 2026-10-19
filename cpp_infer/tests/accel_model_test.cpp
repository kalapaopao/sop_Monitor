#define SOP_ACCEL
#define SOP_INFER_EMBEDDED
#include "main.cpp"
extern "C" int sopRegisterSiLU();
namespace {
#include "joint_detector.hpp"
}
int main(int argc,char** argv) {
 try {
    if(argc<3) throw std::runtime_error("usage: accel_model_test ENGINE IMAGE...");
    sopRegisterSiLU();
    auto root=std::make_shared<TrtModel>(argv[1]);
    TrtModel gpu(root); gpu.gpuPreprocess=true;
    std::vector<cv::Mat> samples;
    std::vector<std::vector<Detection>> references;
    double worstScore=0.,worstCorner=0.;
    const bool pairs=std::string(argv[2])=="--decode-pairs";
    for(int i=pairs?3:2;i<argc;++i) {
        cv::Mat image=cv::imread(argv[i]); if(image.empty()) throw std::runtime_error("Cannot read audit image");
        auto cpu=jointDetections(*root,image,.25f), accelerated=jointDetections(gpu,image,.25f);
        samples.push_back(image); references.push_back(accelerated);
        if(cpu.size()!=accelerated.size()) throw std::runtime_error("Detection count mismatch");
        std::vector<bool> used(cpu.size());
        for(const auto& a:accelerated) {
            int best=-1; float overlap=0;
            for(size_t j=0;j<cpu.size();++j) if(!used[j] && cpu[j].label==a.label) {
                const auto v=polygonIou(cpu[j].points,a.points); if(v>overlap) {overlap=v;best=int(j);}
            }
            if(best<0 || overlap<.95) throw std::runtime_error("Box geometry mismatch");
            used[best]=true;
            const double delta=std::abs(a.confidence-cpu[best].confidence); worstScore=std::max(worstScore,delta);
            double corners=0.; for(size_t p=0;p<4;++p) corners=std::max(corners,cv::norm(a.points[p]-cpu[best].points[p]));
            worstCorner=std::max(worstCorner,corners);
            if(delta>.02 || corners>3.) throw std::runtime_error("Model result tolerance exceeded");
        }
        std::cout<<argv[i]<<" detections="<<cpu.size()<<" PASS\n";
    }
    if(samples.empty()) throw std::runtime_error("No audit images");
    if(pairs) {
        if(samples.size()%2) throw std::runtime_error("Expected CPU/hardware image pairs");
        for(size_t i=0;i<references.size();i+=2) {
            const auto& a=references[i]; const auto& b=references[i+1];
            if(a.size()!=b.size()) throw std::runtime_error("Decode detection count mismatch");
            std::vector<bool> used(b.size());
            for(const auto& d:a) {
                int best=-1; float overlap=0;
                for(size_t j=0;j<b.size();++j) if(!used[j] && b[j].label==d.label) {
                    auto v=polygonIou(d.points,b[j].points); if(v>overlap) { best=int(j); overlap=v; }
                }
                if(best<0 || overlap<.95) throw std::runtime_error("Decode box mismatch");
                used[best]=true;
                const double delta=std::abs(d.confidence-b[best].confidence); worstScore=std::max(worstScore,delta);
                double corners=0.; for(size_t p=0;p<4;++p) corners=std::max(corners,cv::norm(d.points[p]-b[best].points[p]));
                worstCorner=std::max(worstCorner,corners);
                std::cout<<"decode_pair="<<i/2<<" class="<<d.label<<" conf_delta="<<delta<<" corner_delta="<<corners<<'\n';
                if(delta>.02 || corners>3.) throw std::runtime_error("Decode model tolerance exceeded");
            }
        }
    }
    // Different images in parallel expose context/buffer cross-contamination.
    std::vector<std::unique_ptr<TrtModel>> contexts;
    for(int i=0;i<4;++i) { contexts.emplace_back(new TrtModel(root)); contexts.back()->gpuPreprocess=true; }
    std::vector<std::thread> threads; std::mutex errorMutex; std::exception_ptr failure;
    for(int i=0;i<4;++i) threads.emplace_back([&,i] {
        try {
            checkCuda(cudaSetDevice(0),"test device");
            const auto slot=size_t(i)%samples.size(); const auto& ref=references[slot];
            for(int repeat=0;repeat<3;++repeat) {
                const auto actual=jointDetections(*contexts[i],samples[slot],.25f);
                if(actual.size()!=ref.size()) throw std::runtime_error("Concurrent context count mismatch");
                for(size_t k=0;k<actual.size();++k) {
                    if(actual[k].label!=ref[k].label || std::abs(actual[k].confidence-ref[k].confidence)>1e-4f)
                        throw std::runtime_error("Concurrent context result mismatch");
                    for(size_t p=0;p<4;++p) if(cv::norm(actual[k].points[p]-ref[k].points[p])>.01)
                        throw std::runtime_error("Concurrent geometry mismatch");
                }
            }
        } catch(...) { std::lock_guard<std::mutex> lock(errorMutex); if(!failure) failure=std::current_exception(); }
    });
    for(auto& thread:threads) thread.join(); if(failure) std::rethrow_exception(failure);
    std::cout<<"PASS four concurrent contexts; max_conf_delta="<<worstScore<<" max_corner_pixels="<<worstCorner<<'\n'; return 0;
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
