#include "accel_preprocess.hpp"
#include <opencv2/opencv.hpp>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

void checked(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int main() {
 try {
    cudaStream_t stream; checked(cudaStreamCreate(&stream));
    double worst=0.;
    for (const auto size : {cv::Size(1920,1080),cv::Size(321,199),cv::Size(193,511),cv::Size(1,1)}) {
        cv::Mat storage(size.height+2,size.width+4,CV_8UC3); cv::randu(storage,0,256);
        cv::Mat src=storage(cv::Rect(1,1,size.width,size.height)); // deliberately non-contiguous
        const int side=640; float ratio=std::min(float(side)/src.cols,float(side)/src.rows);
        const int rw=std::max(1,int(std::round(src.cols*ratio))), rh=std::max(1,int(std::round(src.rows*ratio)));
        const int left=(side-rw)/2, top=(side-rh)/2;
        cv::Mat canvas(side,side,CV_8UC3,cv::Scalar(114,114,114)), resized;
        cv::resize(src,resized,cv::Size(rw,rh)); resized.copyTo(canvas(cv::Rect(left,top,rw,rh)));
        unsigned char* ds=nullptr; void* dd=nullptr; const size_t pitch=size_t(src.cols)*3;
        checked(cudaMalloc(reinterpret_cast<void**>(&ds),pitch*src.rows)); checked(cudaMalloc(&dd,side*side*3*sizeof(float)));
        checked(cudaMemcpy2DAsync(ds,pitch,src.data,src.step,pitch,src.rows,cudaMemcpyHostToDevice,stream));
        for (bool half : {false,true}) {
            checked(sopLetterbox(ds,src.cols,src.rows,pitch,dd,half,side,rw,rh,left,top,stream));
            std::vector<float> got(side*side*3); std::vector<__half> halves(got.size());
            checked(cudaMemcpyAsync(half?static_cast<void*>(halves.data()):static_cast<void*>(got.data()),dd,
                got.size()*(half?sizeof(__half):sizeof(float)),cudaMemcpyDeviceToHost,stream));
            checked(cudaStreamSynchronize(stream));
            double error=0.;
            for(int c=0;c<3;++c) for(int y=0;y<side;++y) for(int x=0;x<side;++x) {
                size_t i=c*side*side+y*side+x;
                float value=half?__half2float(halves[i]):got[i];
                double delta=std::abs(value-canvas.at<cv::Vec3b>(y,x)[2-c]/255.f);
                if(!std::isfinite(value)) throw std::runtime_error("Nonfinite pixel");
                error=std::max(error,delta);
            }
            worst=std::max(worst,error);
            std::cout << size << " half=" << half << " max_abs=" << error << '\n';
            // OpenCV's 8-bit interpolation rounds differently by at most one level.
            if(error>1./255.+0.0005) throw std::runtime_error("Preprocess mismatch");
        }
        checked(cudaFree(ds)); checked(cudaFree(dd));
    }
    checked(cudaStreamDestroy(stream)); std::cout<<"PASS worst="<<worst<<'\n'; return 0;
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
