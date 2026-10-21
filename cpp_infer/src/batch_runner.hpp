struct BatchRun {
    std::vector<ResultPtr> results;
    double sourceMs=0,detectorMs=0,poseMs=0,totalMs=0;
    std::vector<int> detectorSizes,poseSizes;
};
BatchRun inferGroup(TrtModel& detector,TrtModel& poseModel,const std::vector<FramePtr>& frames,
                    int width,int height,int detectorBatch,int poseBatch) {
    if(frames.empty() || frames.size()>4)throw std::runtime_error("Invalid frame group");
    BatchRun run; const auto start=PipelineClock::now();
    std::vector<cv::Mat> images;
    for(const auto& frame:frames) {
        images.push_back(frame->image);
        auto result=std::make_shared<PipelineResult>();result->frame=frame;run.results.push_back(result);
    }
    const auto views=detector.uploadBatch(images);
    run.sourceMs=milliseconds(start,PipelineClock::now());
    struct Person { size_t owner; cv::Rect2f box; cv::Point2f center; cv::Size2f scale; cv::Mat inverse; };
    std::vector<Person> people;
    for(size_t first=0;first<frames.size();first+=detectorBatch) {
        const size_t count=std::min<size_t>(detectorBatch,frames.size()-first);
        std::vector<SopBatchView> inputs(views.begin()+first,views.begin()+first+count);
        const auto& output=detector.inferBatchDetector(inputs,width,height);
        run.detectorSizes.push_back(detector.batchSlots(count));
        if(output.size()!=1)throw std::runtime_error("Expected one raw OBB tensor");
        for(size_t k=0;k<count;++k) {
            const size_t owner=first+k;auto& result=*run.results[owner];
            result.detections=decodeJointBatchItem(output[0],k,frames[owner]->image.size(),width,height);
            const auto boxes=jointPeople(result.detections,frames[owner]->image.size());
            result.personCalls=boxes.size();result.poseCalls=boxes.size();result.obbCalls=1;
            for(const auto& box:boxes) {
                Person person{};person.owner=owner;person.box=box;
                const auto affine=poseAffine(box,person.center,person.scale);
                cv::invertAffineTransform(affine,person.inverse);people.push_back(person);
            }
        }
    }
    const auto poseStarted=PipelineClock::now();run.detectorMs=milliseconds(start,poseStarted);
    std::vector<float> best(frames.size(),-1.f);
    for(size_t first=0;first<people.size();first+=poseBatch) {
        const size_t count=std::min<size_t>(poseBatch,people.size()-first);
        std::vector<SopBatchView> inputs;std::vector<cv::Mat> inverses;
        for(size_t k=0;k<count;++k) { inputs.push_back(views[people[first+k].owner]);inverses.push_back(people[first+k].inverse); }
        const auto& outputs=poseModel.inferBatchPose(inputs,inverses);run.poseSizes.push_back(poseModel.batchSlots(count));
        for(size_t k=0;k<count;++k) {
            const auto& person=people[first+k];
            const auto pose=decodePoseBatchItem(outputs,k,person.box,person.center,person.scale);
            const float score=std::accumulate(pose.scores.begin()+5,pose.scores.begin()+11,0.f)/6.f;
            if(score>best[person.owner]) {
                best[person.owner]=score;run.results[person.owner]->pose=pose;run.results[person.owner]->hasPose=true;
            }
        }
    }
    run.poseMs=milliseconds(poseStarted,PipelineClock::now());run.totalMs=milliseconds(start,PipelineClock::now());
    for(auto& result:run.results) {
        result->obbFrame=result->frame->id;result->poseFrame=result->hasPose?int64_t(result->frame->id):-1;
        result->obbMs=run.detectorMs/frames.size();result->personPoseMs=run.poseMs/frames.size();
        result->stepMs=run.totalMs;
    }
    return run;
}

void validateBatchSamples(TrtModel& detector,TrtModel& poseModel,const std::string& samples,
                         const std::string& directory,int width,int height,int detectorBatch,int poseBatch) {
    std::vector<cv::String> files;cv::glob(samples+"/*.png",files,false);
    if(files.empty())throw std::runtime_error("No PNG validation samples");
    std::ofstream ds(directory+"/detections.csv"),ps(directory+"/poses.csv"),meta(directory+"/samples.csv");
    if(!ds || !ps || !meta)throw std::runtime_error("Validation directory must exist");
    ds<<"sample,label,confidence,x0,y0,x1,y1,x2,y2,x3,y3\n"<<std::setprecision(9);
    ps<<"sample,keypoint,x,y,confidence\n"<<std::setprecision(9);
    meta<<"sample,person_count,has_pose,width,height,frame_hash\n";
    const size_t groupSize=std::max(detectorBatch,poseBatch);
    for(size_t first=0;first<files.size();first+=groupSize) {
        std::vector<FramePtr> frames;const size_t count=std::min(groupSize,files.size()-first);
        for(size_t i=0;i<count;++i) {
            auto frame=std::make_shared<PipelineFrame>();frame->id=first+i;
            frame->image=cv::imread(files[first+i]);frame->captured=PipelineClock::now();frames.push_back(frame);
        }
        const auto run=inferGroup(detector,poseModel,frames,width,height,detectorBatch,poseBatch);
        for(size_t i=0;i<count;++i) {
            const std::string path=files[first+i],name=path.substr(path.find_last_of("/\\")+1);
            const auto& result=*run.results[i];
            meta<<name<<','<<result.personCalls<<','<<result.hasPose<<','<<width<<','<<height<<','<<resultFingerprint(result)<<'\n';
            for(const auto& d:result.detections) {
                ds<<name<<','<<d.label<<','<<d.confidence;for(const auto& point:d.points)ds<<','<<point.x<<','<<point.y;ds<<'\n';
            }
            if(result.hasPose)for(size_t k=0;k<result.pose.keypoints.size();++k)
                ps<<name<<','<<k<<','<<result.pose.keypoints[k].x<<','<<result.pose.keypoints[k].y<<','<<result.pose.scores[k]<<'\n';
            cv::Mat overlay;cv::resize(frames[i]->image,overlay,cv::Size(960,540));
            auto detections=result.detections;auto pose=result.pose;
            for(auto& d:detections)for(auto& point:d.points)point*=.5f;
            for(auto& point:pose.keypoints)point*=.5f;
            pose.bbox.x*=.5f;pose.bbox.y*=.5f;pose.bbox.width*=.5f;pose.bbox.height*=.5f;
            draw(overlay,detections,result.hasPose?&pose:nullptr,0,run.totalMs);
            if(!cv::imwrite(directory+"/"+name+".jpg",overlay))throw std::runtime_error("Cannot write validation overlay");
        }
    }
    std::cout<<"validation=complete samples="<<files.size()<<" detector_batch="<<detectorBatch<<" pose_batch="<<poseBatch<<'\n';
}
