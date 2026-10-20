#define SOP_STAGE2
#define SOP_ACCEL
#define SOP_INFER_EMBEDDED
#include "main.cpp"
#include "hardware_capture.hpp"
extern "C" int sopRegisterSiLU();
namespace {
#include "joint_detector.hpp"
}
void warpNumerics() {
    cv::Mat storage(513,713,CV_8UC3); cv::randu(storage,0,256);
    cv::Mat image=storage(cv::Rect(3,2,701,503));
    void *source=nullptr,*dest=nullptr; cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream),"test stream");
    checkCuda(cudaMalloc(&source,image.total()*3),"test input");
    checkCuda(cudaMalloc(&dest,192*256*3*sizeof(float)),"test output");
    checkCuda(cudaMemcpy2DAsync(source,image.cols*3,image.data,image.step,image.cols*3,image.rows,cudaMemcpyHostToDevice,stream),"test source copy");
    double worst=0.;
    for(const auto& box: {cv::Rect2f(30,20,380,430),cv::Rect2f(-100,-70,270,350),
                         cv::Rect2f(650,450,120,160),cv::Rect2f(1.25,15.75,301.5,240.1)}) {
        cv::Point2f center; cv::Size2f scale; auto crop=poseWarp(image,box,center,scale);
        const auto expected=chwFloat(crop,true,cv::Scalar(123.675,116.28,103.53),cv::Scalar(58.395,57.12,57.375));
        cv::Mat inverse; cv::invertAffineTransform(poseAffine(box,center,scale),inverse);
        SopInverseAffine matrix{}; for(int r=0;r<2;++r) for(int c=0;c<3;++c) matrix.m[r*3+c]=inverse.at<double>(r,c);
        for(bool half:{false,true}) {
            checkCuda(sopPoseWarp(static_cast<unsigned char*>(source),image.cols,image.rows,image.cols*3,dest,half,matrix,stream),"test warp");
            std::vector<float> actual(expected.size()); std::vector<uint16_t> packed(expected.size());
            checkCuda(cudaMemcpyAsync(half?static_cast<void*>(packed.data()):static_cast<void*>(actual.data()),dest,
                     actual.size()*(half?sizeof(uint16_t):sizeof(float)),cudaMemcpyDeviceToHost,stream),"test output copy");
            checkCuda(cudaStreamSynchronize(stream),"test wait");
            double error=0.;
            for(size_t i=0;i<actual.size();++i) {
                float got=half?halfToFloat(packed[i]):actual[i];
                if(!std::isfinite(got)) throw std::runtime_error("Nonfinite pose tensor");
                error=std::max(error,double(std::abs(got-expected[i])));
            }
            worst=std::max(worst,error); std::cout<<"warp box="<<box<<" half="<<half<<" max_abs="<<error<<'\n';
            if(error>(half?.002:1e-5)) throw std::runtime_error("RTMW affine/normalization mismatch");
        }
    }
    checkCuda(cudaFree(source),"free test input"); checkCuda(cudaFree(dest),"free test output");
    checkCuda(cudaStreamDestroy(stream),"destroy test stream"); std::cout<<"PASS warp worst="<<worst<<'\n';
}
int main(int argc,char** argv) {
 try {
    warpNumerics(); if(argc==1) return 0;
    if(argc<4) throw std::runtime_error("usage: stage2_test JOINT_PLAN RTMW_PLAN [--i420] VIDEO...");
    const bool native=std::string(argv[3])=="--i420";
    sopRegisterSiLU(); auto reference=std::make_shared<TrtModel>(argv[1]); TrtModel detector(reference);
    detector.gpuPreprocess=true; auto poseReference=std::make_shared<TrtModel>(argv[2]); TrtModel pose(poseReference);
    double maxConfidence=0.,maxCorner=0.,maxPose=0.,maxPoseScore=0.;
    for(int v=native?4:3;v<argc;++v) {
        AcceleratedCapture capture; capture.open(argv[v],"gst-h265"); capture.keepNative=true;
        for(int frame=0;frame<=60;++frame) {
            cv::Mat image; if(!capture.read(image)) throw std::runtime_error("Cannot read test frame");
            if(frame%30) continue;
            auto baseline=jointDetections(*reference,image,.25f);
            auto color=capture.sourceColor();
            detector.beginFrame(image,native?capture.nativeI420():cv::Mat(),color.matrix,color.fullRange);
            auto actual=jointDetections(detector,image,.25f);
            if(native) {
                cv::Mat converted(image.size(),CV_8UC3),delta;
                checkCuda(cudaMemcpy(converted.data,detector.sourcePixels(),image.total()*3,cudaMemcpyDeviceToHost),"test native pixels");
                cv::absdiff(image,converted,delta); double maxPixel=0.; cv::minMaxLoc(delta.reshape(1),nullptr,&maxPixel);
                const auto mean=cv::mean(delta);
                std::cout<<"i420_pixel_mae="<<(mean[0]+mean[1]+mean[2])/3.<<" max_pixel="<<maxPixel<<'\n';
            }
            if(actual.size()!=baseline.size()) throw std::runtime_error("Detection count mismatch");
            std::vector<bool> used(baseline.size());
            for(const auto& a:actual) {
                int best=-1; float overlap=0;
                for(size_t j=0;j<baseline.size();++j) if(!used[j] && baseline[j].label==a.label) {
                    const float value=polygonIou(a.points,baseline[j].points); if(value>overlap) {best=int(j);overlap=value;}
                }
                if(best<0 || overlap<.95) throw std::runtime_error("Detection geometry mismatch");
                used[best]=true; double corners=0;
                for(int p=0;p<4;++p) corners=std::max(corners,cv::norm(a.points[p]-baseline[best].points[p]));
                const double conf=std::abs(a.confidence-baseline[best].confidence);
                maxConfidence=std::max(maxConfidence,conf); maxCorner=std::max(maxCorner,corners);
                std::cout<<"detection class="<<a.label<<" conf_delta="<<conf<<" corner_delta="<<corners<<'\n';
                if(conf>.02 || corners>3.) throw std::runtime_error("Detection parity tolerance exceeded");
            }
            for(const auto& box:jointPeople(baseline,image.size())) {
                const auto cpu=rtmwPose(*poseReference,image,box), gpu=rtmwPoseGpu(pose,detector,image,box);
                double pointDelta=0.,scoreDelta=0.;
                for(size_t k=0;k<cpu.scores.size();++k) {
                    scoreDelta=std::max(scoreDelta,double(std::abs(cpu.scores[k]-gpu.scores[k])));
                    if(cpu.scores[k]>=.35 && gpu.scores[k]>=.35) pointDelta=std::max(pointDelta,cv::norm(cpu.keypoints[k]-gpu.keypoints[k]));
                }
                maxPose=std::max(maxPose,pointDelta); maxPoseScore=std::max(maxPoseScore,scoreDelta);
                std::cout<<"pose point_delta="<<pointDelta<<" score_delta="<<scoreDelta<<'\n';
                if(pointDelta>3. || scoreDelta>.02) throw std::runtime_error("Pose parity tolerance exceeded");
            }
            std::cout<<"PASS video="<<v<<" frame="<<frame<<" native="<<native<<'\n';
        }
    }
    detector.reportIO("test_detector"); pose.reportIO("test_pose");
    std::cout<<"PASS model max_conf="<<maxConfidence<<" max_corner="<<maxCorner<<" max_pose="<<maxPose<<" max_pose_score="<<maxPoseScore<<'\n';
    return 0;
 } catch(const std::exception& e) { std::cerr<<"FAILED "<<e.what()<<'\n'; return 1; }
}
