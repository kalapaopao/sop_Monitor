#include "preprocess/batch_kernels.hpp"
#include "preprocess/accel_preprocess.hpp"
#include <vector>
#include <cstring>
#include <iostream>
#include <stdexcept>
void check(cudaError_t value) { if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value)); }
int main() {
    try {
        for(bool half:{false,true}) {
            const size_t unit=half?2:4;
            const int w=1920,h=1080;
            std::vector<unsigned char> image(size_t(w)*h*3);
            for(size_t i=0;i<image.size();++i)image[i]=(i*17+11)%251;
            unsigned char *source=nullptr,*reference=nullptr,*square=nullptr,*rect=nullptr;
            check(cudaMalloc(reinterpret_cast<void**>(&source),image.size()));
            check(cudaMalloc(reinterpret_cast<void**>(&reference),size_t(3)*640*640*unit));
            check(cudaMalloc(reinterpret_cast<void**>(&square),size_t(3)*640*640*unit));
            check(cudaMalloc(reinterpret_cast<void**>(&rect),size_t(2)*3*640*384*unit));
            check(cudaMemcpy(source,image.data(),image.size(),cudaMemcpyHostToDevice));
            cudaStream_t stream;check(cudaStreamCreate(&stream));
            check(sopLetterbox(source,w,h,size_t(w)*3,reference,half,640,640,360,0,140,stream));
            SopBatchView view{source,w,h,size_t(w)*3};
            check(sopRectLetterbox(view,square,half,640,640,stream));
            check(sopRectLetterbox(view,rect,half,640,384,stream));
            check(sopRectLetterbox(view,rect+size_t(3)*640*384*unit,half,640,384,stream));
            check(cudaStreamSynchronize(stream));
            std::vector<unsigned char> a(size_t(3)*640*640*unit),b(a.size()),c(size_t(2)*3*640*384*unit);
            check(cudaMemcpy(a.data(),reference,a.size(),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(b.data(),square,b.size(),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(c.data(),rect,c.size(),cudaMemcpyDeviceToHost));
            if(a!=b)throw std::runtime_error("Square preprocessing changed");
            for(int channel=0;channel<3;++channel)for(int y=0;y<360;++y) {
                const auto* old=a.data()+(size_t(channel)*640*640+(y+140)*640)*unit;
                const auto* current=c.data()+(size_t(channel)*640*384+(y+12)*640)*unit;
                if(std::memcmp(old,current,640*unit))throw std::runtime_error("Rectangular content pixels changed");
            }
            if(std::memcmp(c.data(),c.data()+size_t(3)*640*384*unit,size_t(3)*640*384*unit))
                throw std::runtime_error("Batch item offset changed content");
            check(cudaStreamDestroy(stream));cudaFree(source);cudaFree(reference);cudaFree(square);cudaFree(rect);
        }
        std::cout<<"FP32/FP16 square parity, rectangular content and batch offsets passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
