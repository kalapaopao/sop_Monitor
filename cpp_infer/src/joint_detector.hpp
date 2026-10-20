// Included in the existing anonymous namespace.
std::vector<Detection> jointDetections(TrtModel& model, const cv::Mat& frame, float threshold) {
    float ratio=std::min(640.f/frame.cols,640.f/frame.rows);
    int left=(640-int(std::round(frame.cols*ratio)))/2, top=(640-int(std::round(frame.rows*ratio)))/2;
#ifdef SOP_STAGE2
    const auto& outputs = [&]() -> ModelOutputs {
        if(model.gpuPreprocess) return model.inferLetterbox(frame);
        const cv::Mat canvas=letterboxCentered(frame,640,ratio,left,top);
        cv::Mat rgb; cv::cvtColor(canvas,rgb,cv::COLOR_BGR2RGB);
        auto input=chwFloat(rgb,true,cv::Scalar(),cv::Scalar(255,255,255));
        return model.infer(input,nvinfer1::Dims4{1,3,640,640});
    }();
#else
    std::vector<TensorResult> outputs;
#ifdef SOP_ACCEL
    if (model.gpuPreprocess) outputs = model.inferLetterbox(frame);
    else
#endif
    {
        const cv::Mat canvas = letterboxCentered(frame, 640, ratio, left, top);
        cv::Mat rgb; cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
        auto input = chwFloat(rgb, true, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        outputs = model.infer(input, nvinfer1::Dims4{1, 3, 640, 640});
    }
#endif
    if (outputs.empty() || outputs[0].size() != 8400u * 9u) throw std::runtime_error("Unexpected YOLO raw output shape");
    const auto& raw = outputs[0]; const std::array<std::string, 4> labels{"screwdriver1", "screwdriver2", "hand", "person"};
    struct Candidate { int anchor; int cls; float score; };
    std::vector<std::pair<float, int>> best; best.reserve(8400);
    for (int i = 0; i < 8400; ++i) best.emplace_back(std::max({raw.value(i * 9 + 4), raw.value(i * 9 + 5), raw.value(i * 9 + 6), raw.value(i * 9 + 7)}), i);
    const size_t keepAnchors = std::min<size_t>(300, best.size()); std::partial_sort(best.begin(), best.begin() + keepAnchors, best.end(), std::greater<>());
    std::vector<Candidate> candidates;
    for (size_t k = 0; k < keepAnchors; ++k) { const int i = best[k].second; for (int cls = 0; cls < 4; ++cls) if (raw.value(i * 9 + 4 + cls) >= threshold) candidates.push_back({i, cls, raw.value(i * 9 + 4 + cls)}); }
    if (candidates.size() > 300) { std::partial_sort(candidates.begin(), candidates.begin() + 300, candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; }); candidates.resize(300); }
    std::vector<Detection> detections;
    for (const auto& c : candidates) {
        const float cx = (raw.value(c.anchor * 9) - left) / ratio, cy = (raw.value(c.anchor * 9 + 1) - top) / ratio, w = raw.value(c.anchor * 9 + 2) / ratio, h = raw.value(c.anchor * 9 + 3) / ratio;
        cv::RotatedRect rect(cv::Point2f(cx, cy), cv::Size2f(w, h), raw.value(c.anchor * 9 + 8) * 180.0f / static_cast<float>(CV_PI));
        std::array<cv::Point2f, 4> points; rect.points(points.data()); detections.push_back({labels[c.cls], c.score, rect, points});
    }
    std::vector<Detection> kept;
    for (const auto& label : labels) {
        std::vector<Detection> same; for (const auto& d : detections) if (d.label == label) same.push_back(d);
        std::sort(same.begin(), same.end(), [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
        for (const auto& d : same) { bool accept = true; for (const auto& prior : kept) if (prior.label == d.label && polygonIou(prior.points, d.points) >= 0.55f) { accept = false; break; } if (accept) kept.push_back(d); }
    }
    return kept;
}


std::vector<cv::Rect2f> jointPeople(const std::vector<Detection>& detections, cv::Size size) {
    std::vector<cv::Rect2f> boxes;
    const cv::Rect2f bounds(0,0,static_cast<float>(size.width),static_cast<float>(size.height));
    for (const auto& d : detections) if (d.label == "person") {
        // RTMW expects a horizontal person ROI; retain the complete rotated person.
        float x0 = d.points[0].x, y0 = d.points[0].y, x1 = x0, y1 = y0;
        for (const auto& p : d.points) { x0=std::min(x0,p.x); y0=std::min(y0,p.y); x1=std::max(x1,p.x); y1=std::max(y1,p.y); }
        auto box = cv::Rect2f(x0,y0,x1-x0,y1-y0) & bounds;
        if (box.width > 1 && box.height > 1) boxes.push_back(box);
    }
    return boxes;
}
