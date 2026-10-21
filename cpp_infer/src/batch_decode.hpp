// Same postprocessing arithmetic as the existing single-image runner.
struct TensorBatchSlice {
    const TensorResult& tensor; size_t offset,count;
    float value(size_t i) const { if(i>=count)throw std::out_of_range("Batch tensor slice"); return tensor.value(offset+i); }
    size_t size() const { return count; }
    nvinfer1::DataType type= tensor.type;
};
std::vector<Detection> decodeJointBatchItem(const TensorResult& tensor,size_t index,cv::Size original,int width,int height,float threshold=.25f) {
    if(tensor.dims.nbDims!=3 || tensor.dims.d[1]<=0 || tensor.dims.d[2]!=9 || index>=size_t(tensor.dims.d[0]))
        throw std::runtime_error("Unexpected batch OBB output");
    const size_t anchors=tensor.dims.d[1];
    const float ratio=std::min(float(width)/original.width,float(height)/original.height);
    const int left=(width-int(std::round(original.width*ratio)))/2;
    const int top=(height-int(std::round(original.height*ratio)))/2;
    TensorBatchSlice raw{tensor,index*anchors*9,anchors*9}; const std::array<std::string, 4> labels{"screwdriver1", "screwdriver2", "hand", "person"};
    struct Candidate { int anchor; int cls; float score; };
    std::vector<std::pair<float, int>> best; best.reserve(anchors);
    for (size_t i = 0; i < anchors; ++i) best.emplace_back(std::max({raw.value(i * 9 + 4), raw.value(i * 9 + 5), raw.value(i * 9 + 6), raw.value(i * 9 + 7)}), i);
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
PoseResult decodePoseBatchItem(const std::vector<TensorResult>& outputs,size_t index,const cv::Rect2f& box,
    const cv::Point2f& center,const cv::Size2f& scale) {

    if(outputs.size()!=2 || outputs[0].dims.nbDims!=3 || outputs[1].dims.nbDims!=3 || outputs[0].dims.d[1]!=outputs[1].dims.d[1] || outputs[0].dims.d[2]!=384 || outputs[1].dims.d[2]!=512 ||
        index>=size_t(outputs[0].dims.d[0]) || outputs[0].dims.d[0]!=outputs[1].dims.d[0])
        throw std::runtime_error("Unexpected batch RTMW outputs");
    const size_t count=outputs[0].dims.d[1];
    TensorBatchSlice sx{outputs[0],index*count*384,count*384};
    TensorBatchSlice sy{outputs[1],index*count*512,count*512}; const size_t keypointCount = sx.size() / 384u; if (keypointCount == 0 || keypointCount > 134 || sx.size() != keypointCount * 384u || sy.size() != keypointCount * 512u) { std::cerr << "[rtmw] output sizes " << sx.size() << "," << sy.size() << " types " << static_cast<int>(sx.type) << "," << static_cast<int>(sy.type) << "\n"; throw std::runtime_error("Unexpected RTMW output shape"); } PoseResult result{}; result.bbox = box; result.valid = true;
    for (size_t k = 0; k < keypointCount; ++k) { size_t ix = 0, iy = 0; float maxx = -std::numeric_limits<float>::infinity(), maxy = -std::numeric_limits<float>::infinity(); for (size_t x = 0; x < 384; ++x) if (sx.value(k * 384 + x) > maxx) maxx = sx.value(k * 384 + x), ix = x; for (size_t y = 0; y < 512; ++y) if (sy.value(k * 512 + y) > maxy) maxy = sy.value(k * 512 + y), iy = y; result.scores[k] = (maxx + maxy) * .5f; const float px = static_cast<float>(ix) / 2.0f, py = static_cast<float>(iy) / 2.0f; result.keypoints[k] = cv::Point2f(px / 192.0f * scale.width + center.x - scale.width * .5f, py / 256.0f * scale.height + center.y - scale.height * .5f); }
    return result;
}
