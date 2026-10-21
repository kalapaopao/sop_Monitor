    std::vector<SopBatchView> uploadBatch(const std::vector<cv::Mat>& images) {
        size_t total=0;
        for(const auto& image:images) {
            if(image.empty() || image.type()!=CV_8UC3)throw std::runtime_error("Batch source must be BGR8");
            total+=image.total()*3;
        }
        if(total>sourceBytes_) {
            if(sourceDevice_)checkCuda(cudaFree(sourceDevice_),"free batch sources");
            sourceDevice_=nullptr;sourceBytes_=0;
            checkCuda(cudaMalloc(&sourceDevice_,total),"allocate batch sources");sourceBytes_=total;
        }
        auto* pinned=static_cast<unsigned char*>(io_.source.ensure(total));
        std::vector<SopBatchView> views;size_t offset=0;
        for(const auto& image:images) {
            const size_t row=size_t(image.cols)*3;
            for(int y=0;y<image.rows;++y)std::memcpy(pinned+offset+row*y,image.ptr(y),row);
            views.push_back({static_cast<unsigned char*>(sourceDevice_)+offset,image.cols,image.rows,row});
            offset+=image.total()*3;
        }
        checkCuda(cudaMemcpyAsync(sourceDevice_,pinned,total,cudaMemcpyHostToDevice,stream_),"upload batch originals");
        io_.uploadBytes+=total;
        return views;
    }
    int batchSlots(size_t requested) const {
        const int fixed=engine_->getBindingDimensions(inputIndex_).d[0];
        if(fixed>0 && requested>size_t(fixed))throw std::runtime_error("Requested batch exceeds fixed engine");
        return fixed>0?fixed:int(requested);
    }
    ModelOutputs inferBatchDetector(const std::vector<SopBatchView>& images,int width,int height) {
        if(images.empty() || images.size()>4)throw std::runtime_error("Invalid detector batch");
        const int slots=batchSlots(images.size());
        if(!context_->setBindingDimensions(inputIndex_,nvinfer1::Dims4{slots,3,height,width}))
            throw std::runtime_error("Engine rejected detector shape");
        resizeBuffers();
        const auto type=engine_->getBindingDataType(inputIndex_);
        if(type!=nvinfer1::DataType::kFLOAT && type!=nvinfer1::DataType::kHALF)throw std::runtime_error("Detector input type");
        const size_t bytes=size_t(3)*width*height*typeSize(type);
        for(int i=0;i<slots;++i)
            checkCuda(sopRectLetterbox(images[std::min<size_t>(i,images.size()-1)],static_cast<unsigned char*>(deviceBuffers_[inputIndex_])+i*bytes,
                         type==nvinfer1::DataType::kHALF,width,height,stream_),"batch letterbox");
        return execute();
    }
    ModelOutputs inferBatchPose(const std::vector<SopBatchView>& images,const std::vector<cv::Mat>& inverses) {
        if(images.empty() || images.size()>4 || images.size()!=inverses.size())throw std::runtime_error("Invalid pose batch");
        const int slots=batchSlots(images.size());
        if(!context_->setBindingDimensions(inputIndex_,nvinfer1::Dims4{slots,3,256,192}))
            throw std::runtime_error("Engine rejected pose shape");
        resizeBuffers();
        const auto type=engine_->getBindingDataType(inputIndex_);
        if(type!=nvinfer1::DataType::kFLOAT && type!=nvinfer1::DataType::kHALF)throw std::runtime_error("Pose input type");
        const size_t bytes=size_t(3)*256*192*typeSize(type);
        for(int i=0;i<slots;++i) {
            const size_t input=std::min<size_t>(i,images.size()-1);
            SopInverseAffine affine{};
            for(int r=0;r<2;++r)for(int c=0;c<3;++c)affine.m[r*3+c]=inverses[input].at<double>(r,c);
            checkCuda(sopPoseWarp(images[input].pixels,images[input].width,images[input].height,images[input].pitch,
                       static_cast<unsigned char*>(deviceBuffers_[inputIndex_])+i*bytes,
                       type==nvinfer1::DataType::kHALF,affine,stream_),"batch pose warp");
        }
        return execute();
    }
