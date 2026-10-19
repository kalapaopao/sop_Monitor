#include "hardware_capture.hpp"
int main(int argc,char** argv) {
 try {
    if(argc!=3 && argc!=4) throw std::runtime_error("usage: accel_decode_test VIDEO EXISTING_OUTPUT_DIR [--seek]");
    cv::VideoCapture cpu(argv[1]); AcceleratedCapture hw; hw.open(argv[1],"gst-h265");
    if(!cpu.isOpened()) throw std::runtime_error("Cannot open CPU reference");
    std::cout<<"source="<<hw.get(3)<<'x'<<hw.get(4)<<" fps="<<hw.get(5)<<'\n';
    double worst=0.,worstMean=0.;
    for(int i=0;i<=60;++i) {
        cv::Mat a,b,d;
        if(!cpu.read(a) || !hw.read(b) || a.size()!=b.size() || a.type()!=b.type())
            throw std::runtime_error("Decode frame/layout mismatch");
        cv::absdiff(a,b,d); double maxValue; cv::minMaxLoc(d.reshape(1),nullptr,&maxValue);
        const auto mean=cv::mean(d); const double avg=(mean[0]+mean[1]+mean[2])/3.;
        worst=std::max(worst,maxValue); worstMean=std::max(worstMean,avg);
        if(i%30==0) {
            std::cout<<"frame="<<i<<" MAE="<<avg<<" max="<<maxValue<<'\n';
            const auto base=std::string(argv[2])+"/frame"+std::to_string(i);
            if(!cv::imwrite(base+"_cpu.png",a) || !cv::imwrite(base+"_hw.png",b))
                throw std::runtime_error("Cannot save decode audit");
        }
    }
    std::cout<<"worst_MAE="<<worstMean<<" worst_max="<<worst<<'\n';
    if(worst>1) throw std::runtime_error("Hardware decoding pixel equivalence failed");
    if(argc==4) {
        if(std::string(argv[3])!="--seek") throw std::runtime_error("Unknown test option");
        if(!hw.set(cv::CAP_PROP_POS_MSEC,180000)) throw std::runtime_error("Hardware seek failed");
        cv::Mat sought; if(!hw.read(sought)) throw std::runtime_error("Hardware seek read failed");
    }
    std::cout<<"PASS decode pixel comparison\n"; return 0;
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
