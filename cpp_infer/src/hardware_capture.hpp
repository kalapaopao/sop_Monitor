#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <stdexcept>
#include <iostream>
extern "C" {
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}
struct CaptureColor { int matrix=SWS_CS_ITU601; int fullRange=0; };
inline CaptureColor fileColor(const std::string& source) {
    AVFormatContext* format=nullptr;
    if(avformat_open_input(&format,source.c_str(),nullptr,nullptr)<0)
        throw std::runtime_error("Cannot inspect file color metadata");
    struct Close { AVFormatContext*& ptr; ~Close() { avformat_close_input(&ptr); } } close{format};
    if(avformat_find_stream_info(format,nullptr)<0) throw std::runtime_error("Cannot inspect video stream");
    for(unsigned i=0;i<format->nb_streams;++i) {
        const auto* par=format->streams[i]->codecpar;
        if(par->codec_type!=AVMEDIA_TYPE_VIDEO) continue;
        CaptureColor color;
        if(par->color_space==AVCOL_SPC_BT709) color.matrix=SWS_CS_ITU709;
        else if(par->color_space==AVCOL_SPC_BT2020_NCL) color.matrix=SWS_CS_BT2020;
        else if(par->color_space!=AVCOL_SPC_UNSPECIFIED && par->color_space!=AVCOL_SPC_BT470BG &&
                par->color_space!=AVCOL_SPC_SMPTE170M)
            throw std::runtime_error("Unsupported color matrix; use --decoder opencv");
        color.fullRange=par->color_range==AVCOL_RANGE_JPEG;
        std::cout << "file_color_matrix=" << color.matrix << " full_range=" << color.fullRange
                  << " matrix_unspecified=" << (par->color_space==AVCOL_SPC_UNSPECIFIED) << '\n';
        return color;
    }
    throw std::runtime_error("No video stream");
}
inline std::string gstQuoted(const std::string& value) {
    std::string out="\"";
    for(char c:value) { if(c=='\\' || c=='\"') out+='\\'; out+=c; }
    return out+"\"";
}
class AcceleratedCapture {
public:
    ~AcceleratedCapture() { if(converter_) sws_freeContext(converter_); }
    AcceleratedCapture()=default;
    AcceleratedCapture(const AcceleratedCapture&)=delete;
    AcceleratedCapture& operator=(const AcceleratedCapture&)=delete;
    void open(const std::string& source,const std::string& decoder) {
        if(decoder=="opencv") {
            if(!capture_.open(source)) throw std::runtime_error("Cannot open video");
            return;
        }
        if(decoder!="gst-h265") throw std::runtime_error("decoder must be opencv or gst-h265");
        const bool live=source.rfind("rtsp://",0)==0 || source.rfind("rtsps://",0)==0;
        if(!live) color_=fileColor(source);
        else color_.matrix=SWS_CS_ITU709; // Live-camera color negotiation is not field-validated.
        const std::string prefix=live ?
            "rtspsrc location="+gstQuoted(source)+" protocols=tcp latency=100 drop-on-latency=true ! rtph265depay ! " :
            "filesrc location="+gstQuoted(source)+" ! qtdemux ! ";
        // Retain native YUV; convert with the same libswscale path as OpenCV FFmpeg.
        const std::string pipeline=prefix+
            "h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=I420 ! "
            "appsink max-buffers=2 drop="+(live?std::string("true"):std::string("false"))+" sync=false";
        if(!capture_.open(pipeline,cv::CAP_GSTREAMER))
            throw std::runtime_error("H.265 pipeline failed; use --decoder opencv");
        rawYuv_=true;
    }
    bool read(cv::Mat& frame) {
        if(!rawYuv_) return capture_.read(frame);
        cv::Mat raw;
        if(!capture_.read(raw)) return false;
        const int w=int(capture_.get(cv::CAP_PROP_FRAME_WIDTH)), h=int(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
        if(w<=0 || h<=0 || w%2 || h%2 || raw.type()!=CV_8UC1 || raw.cols!=w || raw.rows!=h*3/2)
            throw std::runtime_error("Unexpected I420 frame layout");
        if(!raw.isContinuous()) raw=raw.clone();
        converter_=sws_getCachedContext(converter_,w,h,AV_PIX_FMT_YUV420P,w,h,AV_PIX_FMT_BGR24,
                                       SWS_BICUBIC,nullptr,nullptr,nullptr);
        if(!converter_) throw std::runtime_error("Cannot create YUV converter");
        const int* coeff=sws_getCoefficients(color_.matrix);
        if(sws_setColorspaceDetails(converter_,coeff,color_.fullRange,coeff,1,0,1<<16,1<<16)<0)
            throw std::runtime_error("Cannot configure YUV color conversion");
        const uint8_t* planes[4]={raw.data,raw.data+w*h,raw.data+w*h*5/4,nullptr};
        const int strides[4]={w,w/2,w/2,0};
        frame.create(h,w,CV_8UC3);
        uint8_t* output[4]={frame.data,nullptr,nullptr,nullptr};
        const int outputStrides[4]={int(frame.step),0,0,0};
        if(sws_scale(converter_,planes,strides,0,h,output,outputStrides)!=h)
            throw std::runtime_error("YUV conversion failed");
        return true;
    }
    double get(int property) const { return capture_.get(property); }
    bool set(int property,double value) { return capture_.set(property,value); }
    std::string getBackendName() const { return capture_.getBackendName(); }
private:
    cv::VideoCapture capture_;
    SwsContext* converter_=nullptr;
    CaptureColor color_;
    bool rawYuv_=false;
};
inline void openAcceleratedCapture(AcceleratedCapture& cap,const std::string& source,const std::string& decoder) {
    cap.open(source,decoder);
}

