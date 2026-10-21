// Public TrtModel methods, compiled only for the independent stage-two target.
    bool reuseIO=true;
    const unsigned char* sourcePixels() const { return static_cast<unsigned char*>(sourceDevice_); }
    void beginFrame(const cv::Mat& bgr,const cv::Mat& i420={},int matrix=5,int fullRange=0) {
#ifdef SOP_PROFILE
        sop_profile::CpuScope sourceHost("source_host_submit");
#endif
        const size_t bytes=bgr.total()*3;
        if(bytes>sourceBytes_) {
            if(sourceDevice_) checkCuda(cudaFree(sourceDevice_),"free frame storage");
            sourceDevice_=nullptr; sourceBytes_=0;
            checkCuda(cudaMalloc(&sourceDevice_,bytes),"allocate frame storage"); sourceBytes_=bytes;
        }
        const bool native=!i420.empty();
        const size_t row=native?size_t(bgr.cols):size_t(bgr.cols)*3;
        const int rows=native?bgr.rows*3/2:bgr.rows;
        const cv::Mat& input=native?i420:bgr;
        if(native && (matrix!=5 || fullRange!=0 || input.type()!=CV_8UC1 || input.cols!=bgr.cols || input.rows!=rows))
            throw std::runtime_error("Experimental I420 input requires BT.601 limited-range planar layout");
        const size_t upload=row*rows;
        void* dst=sourceDevice_;
        if(native) {
            if(upload>io_.nativeBytes) {
                if(io_.nativeDevice) checkCuda(cudaFree(io_.nativeDevice),"free YUV storage");
                io_.nativeDevice=nullptr; io_.nativeBytes=0;
                checkCuda(cudaMalloc(&io_.nativeDevice,upload),"allocate YUV storage"); io_.nativeBytes=upload;
            }
            dst=io_.nativeDevice;
        }
        if(reuseIO) {
            auto* host=static_cast<unsigned char*>(io_.source.ensure(upload));
            for(int y=0;y<rows;++y) std::memcpy(host+y*row,input.ptr(y),row);
#ifdef SOP_PROFILE
            sop_profile::GpuScope uploadGpu("source_upload",stream_);
#endif
            checkCuda(cudaMemcpyAsync(dst,host,upload,cudaMemcpyHostToDevice,stream_),"upload pinned source");
        } else {
#ifdef SOP_PROFILE
            sop_profile::GpuScope uploadGpu("source_upload",stream_);
#endif
            checkCuda(cudaMemcpy2DAsync(dst,row,input.data,input.step,row,rows,cudaMemcpyHostToDevice,stream_),"upload source");
        }
        io_.uploadBytes+=upload;
        if(native) checkCuda(sopI420ToBgr(static_cast<unsigned char*>(dst),bgr.cols,bgr.rows,
                           static_cast<unsigned char*>(sourceDevice_),stream_),"convert I420 on GPU");
        io_.prepared=true;
    }
    ModelOutputs inferWarp(const TrtModel& source,const cv::Size& size,const cv::Mat& inverse) {
        if(!context_->setBindingDimensions(inputIndex_,nvinfer1::Dims4{1,3,256,192}))
            throw std::runtime_error("Invalid RTMW GPU dimensions");
        resizeBuffers();
        const auto type=engine_->getBindingDataType(inputIndex_);
        if(type!=nvinfer1::DataType::kFLOAT && type!=nvinfer1::DataType::kHALF)
            throw std::runtime_error("RTMW GPU input type unsupported");
        {
#ifdef SOP_PROFILE
        sop_profile::CpuScope preHost("preprocess_host_submit");
        sop_profile::GpuScope preGpu("preprocess",stream_);
#endif
        SopInverseAffine affine{}; for(int r=0;r<2;++r) for(int c=0;c<3;++c) affine.m[r*3+c]=inverse.at<double>(r,c);
        // Source detection has already completed before CPU person boxes exist.
        checkCuda(sopPoseWarp(source.sourcePixels(),size.width,size.height,size.width*3,
                   deviceBuffers_[inputIndex_],type==nvinfer1::DataType::kHALF,affine,stream_),"RTMW GPU warp");
        }
        return execute();
    }
    ModelOutputs executeStage2() {
        {
#ifdef SOP_PROFILE
        sop_profile::CpuScope enqueueHost("enqueue_host");
        sop_profile::GpuScope engineGpu("engine_span",stream_);
#endif
        if(!context_->enqueueV2(deviceBuffers_.data(),stream_,nullptr)) throw std::runtime_error("TRT enqueue failed");
        }
        if(!reuseIO) io_.results.clear();
        io_.results.resize(outputIndices_.size());
        {
#ifdef SOP_PROFILE
        sop_profile::CpuScope copyHost("readback_host_submit");
        sop_profile::GpuScope copyGpu("readback",stream_);
#endif
        for(size_t k=0;k<outputIndices_.size();++k) {
            const int index=outputIndices_[k]; auto& result=io_.results[k];
            result.dims=context_->getBindingDimensions(index); result.type=engine_->getBindingDataType(index);
            const size_t count=volume(result.dims),bytes=count*typeSize(result.type);
            if(result.type==nvinfer1::DataType::kFLOAT) {
                if(result.floats.capacity()<count) ++io_.resultAllocations;
                result.floats.resize(count);
            } else {
                if(result.ints.capacity()<count) ++io_.resultAllocations;
                result.ints.resize(count);
            }
            void* destination=reuseIO?io_.output[k].ensure(bytes):
                (result.type==nvinfer1::DataType::kFLOAT?static_cast<void*>(result.floats.data()):static_cast<void*>(result.ints.data()));
            checkCuda(cudaMemcpyAsync(destination,deviceBuffers_[index],bytes,cudaMemcpyDeviceToHost,stream_),"stage2 output copy");
        }
        }
        if(reuseIO) {
            if(!io_.completed) checkCuda(cudaEventCreateWithFlags(&io_.completed,cudaEventDisableTiming),"create completion event");
            checkCuda(cudaEventRecord(io_.completed,stream_),"record completion");
            {
#ifdef SOP_PROFILE
            sop_profile::CpuScope waitHost("completion_wait_host");
#endif
            checkCuda(cudaEventSynchronize(io_.completed),"wait for CPU postprocessing inputs");
            }
#ifdef SOP_PROFILE
            sop_profile::CpuScope materializeHost("readback_materialize_host");
#endif
            for(size_t k=0;k<io_.results.size();++k) {
                auto& r=io_.results[k]; const size_t n=volume(r.dims);
                if(r.type==nvinfer1::DataType::kFLOAT) std::memcpy(r.floats.data(),io_.output[k].data,n*sizeof(float));
                else if(r.type==nvinfer1::DataType::kINT32) std::memcpy(r.ints.data(),io_.output[k].data,n*sizeof(int32_t));
                else if(r.type==nvinfer1::DataType::kHALF) for(size_t i=0;i<n;++i) r.ints[i]=static_cast<uint16_t*>(io_.output[k].data)[i];
                else throw std::runtime_error("Unsupported output type");
            }
        } else {
            checkCuda(cudaStreamSynchronize(stream_),"legacy completion wait");
            for(auto& r:io_.results) if(r.type==nvinfer1::DataType::kHALF) {
                const auto* packed=reinterpret_cast<const uint16_t*>(r.ints.data());
                for(size_t i=r.ints.size();i>0;--i) r.ints[i-1]=packed[i-1];
            }
        }
        return io_.results;
    }
    void reportIO(const char* label) const {
        uint64_t pins=io_.source.allocations+io_.input.allocations;
        for(const auto& p:io_.output) pins+=p.allocations;
        std::cout<<"stage2_io="<<label<<" pinned_allocations="<<pins<<" result_allocations="<<io_.resultAllocations
                 <<" upload_bytes="<<io_.uploadBytes<<'\n';
    }
