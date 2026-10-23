// Included after Args and the existing model functions, inside their namespace.
using PipelineClock = std::chrono::steady_clock;
using PipelineTime = PipelineClock::time_point;
double milliseconds(PipelineTime a, PipelineTime b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

class InferenceWorker {
public:
    InferenceWorker() : thread_([this] {
        std::function<void()> task;
        while (tasks_.pop(task)) task();
    }) {}
    ~InferenceWorker() { tasks_.close(); if (thread_.joinable()) thread_.join(); }
    std::future<void> submit(std::function<void()> action) {
        auto task = std::make_shared<std::packaged_task<void()>>([action] {
            checkCuda(cudaSetDevice(0), "worker cudaSetDevice");
            action();
        });
        auto future = task->get_future();
        if (!tasks_.push([task] { (*task)(); })) throw std::runtime_error("Inference worker closed");
        return future;
    }
private:
    BoundedQueue<std::function<void()>> tasks_{1};
    std::thread thread_;
};

struct PipelineFrame {
    uint64_t id = 0;
    cv::Mat image;
#ifdef SOP_STAGE2
    cv::Mat nativeI420;
    int colorMatrix=5, fullRange=0;
#endif
    PipelineTime captured;
    double decodeMs = 0;
#ifdef SOP_PROFILE
    double profileReadMs=0,profileColorMs=0;
#endif
};
using FramePtr = std::shared_ptr<PipelineFrame>;

struct PipelineResult {
    FramePtr frame;
    std::vector<Detection> detections;
    PoseResult pose;
    bool hasPose = false;
    int64_t obbFrame = -1, poseFrame = -1;
    uint64_t obbCalls = 0, personCalls = 0, poseCalls = 0;
    double stepMs = 0, obbMs = 0, personPoseMs = 0;
};
using ResultPtr = std::shared_ptr<PipelineResult>;

// Hash decoded results, not timing or overlays, to check cross-mode consistency.
uint64_t resultFingerprint(const PipelineResult& result) {
    uint64_t hash = 14695981039346656037ull;
    auto add = [&](float value) {
        uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 4; ++i) { hash ^= (bits >> (i * 8)) & 255u; hash *= 1099511628211ull; }
    };
    add(static_cast<float>(result.detections.size()));
    for (const auto& d : result.detections) {
        for (unsigned char c : d.label) { hash ^= c; hash *= 1099511628211ull; }
        add(d.confidence);
        for (const auto& p : d.points) { add(p.x); add(p.y); }
    }
    add(result.hasPose ? 1.f : 0.f);
    if (result.hasPose) {
        add(result.pose.bbox.x); add(result.pose.bbox.y);
        add(result.pose.bbox.width); add(result.pose.bbox.height);
        for (size_t k = 0; k < result.pose.keypoints.size(); ++k) {
            add(result.pose.keypoints[k].x); add(result.pose.keypoints[k].y); add(result.pose.scores[k]);
        }
    }
    return hash;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(fraction * (values.size() - 1))];
}

#ifndef SOP_BATCH  // runPipeline needs the shared Args; batch_probe has its own
int runPipeline(const Args& args) {
    const bool threaded = args.mode != "serial";
    const bool parallel = args.mode == "parallel";
    checkCuda(cudaSetDevice(0), "pipeline cudaSetDevice");
    // Each model/context/stream has exactly one caller at a time.
    TrtModel yolo(args.yoloEngine), yolox(args.yoloxEngine), rtmw(args.rtmwEngine);
    cv::VideoCapture capture(args.source);
    if (!capture.isOpened()) throw std::runtime_error("Cannot open source: " + args.source);
    const double fps = capture.get(cv::CAP_PROP_FPS) > 0 ? capture.get(cv::CAP_PROP_FPS) : 25.;
    if (args.startSec > 0 && !capture.set(cv::CAP_PROP_POS_MSEC, args.startSec * 1000.))
        throw std::runtime_error("Cannot seek to requested start time");
    std::ofstream csv;
    if (!args.metrics.empty()) {
        csv.open(args.metrics);
        if (!csv) throw std::runtime_error("Cannot create metrics file: " + args.metrics);
        csv << "frame,source_ms,result_frame,obb_frame,pose_frame,decode_ms,step_ms,obb_ms,person_pose_ms,display_gap_ms,capture_to_display_ms,obb_age_ms,pose_age_ms,obb_calls,person_calls,pose_calls,result_hash\n";
    }
    std::unique_ptr<InferenceWorker> obbWorker, poseWorker;
    if (parallel) { obbWorker.reset(new InferenceWorker); poseWorker.reset(new InferenceWorker); }
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    PoseResult bestPose;
    float bestScore = -1;
    int64_t obbFrame = -1, poseFrame = -1;
    uint64_t obbCalls = 0, personCalls = 0, poseCalls = 0;
    std::atomic<uint64_t> decoded{0}, inferred{0}, droppedJobs{0}, droppedPreviews{0};
    std::atomic<bool> stopped{false};
    BoundedQueue<FramePtr> inputs(args.realtime ? 1 : 2), previews(1);
    BoundedQueue<ResultPtr> results(2);
    std::mutex resultMutex, errorMutex;
    ResultPtr latestResult;
    std::exception_ptr error;
    std::vector<std::thread> threads;
    auto stop = [&] { stopped = true; inputs.cancel(); previews.cancel(); results.cancel(); };
    auto fail = [&] {
        { std::lock_guard<std::mutex> lock(errorMutex); if (!error) error = std::current_exception(); }
        stop();
    };
    // Join even when drawing, writing, or worker construction throws.
    struct JoinGuard {
        std::function<void()> stop;
        std::vector<std::thread>& threads;
        void finish() { stop(); for (auto& t : threads) if (t.joinable()) t.join(); }
        ~JoinGuard() { finish(); }
    } guard{stop, threads};

    auto process = [&](const FramePtr& frame) {
        auto result = std::make_shared<PipelineResult>();
        result->frame = frame;
        const auto started = PipelineClock::now();
        // Offline schedules match the legacy modulo schedule exactly.
        // Live jobs can skip frames, so use source-frame distance to avoid aliasing.
        const bool updateObb = args.realtime ? (obbFrame < 0 || static_cast<int64_t>(frame->id) - obbFrame >= args.obbStride)
            : frame->id % args.obbStride == 0;
        auto runObb = [&] {
            const auto t = PipelineClock::now();
            if (updateObb) { detections = yoloDetections(yolo, frame->image, .25f); ++obbCalls; obbFrame = static_cast<int64_t>(frame->id); }
            result->obbMs = milliseconds(t, PipelineClock::now());
        };
        auto runPose = [&] {
            const auto t = PipelineClock::now();
            // In live mode refresh person boxes for every selected pose job.
            // The live inference scheduler already skips unstarted jobs.
            if (args.realtime || frame->id % args.personStride == 0) { boxes = yoloXPeople(yolox, frame->image); ++personCalls; }
            if (boxes.empty()) { bestScore = -1; poseFrame = -1; }
            else if (args.realtime ? (poseFrame < 0 || static_cast<int64_t>(frame->id) - poseFrame >= args.poseStride)
                                  : frame->id % args.poseStride == 0) {
                bestScore = -1;
                for (const auto& box : boxes) {
                    auto pose = rtmwPose(rtmw, frame->image, box); ++poseCalls;
                    const float score = std::accumulate(pose.scores.begin() + 5, pose.scores.begin() + 11, 0.f) / 6.f;
                    if (score > bestScore) { bestScore = score; bestPose = pose; }
                }
                poseFrame = static_cast<int64_t>(frame->id);
            }
            result->personPoseMs = milliseconds(t, PipelineClock::now());
        };
        if (parallel) {
            auto obbDone = obbWorker->submit(runObb);
            std::future<void> poseDone;
            try { poseDone = poseWorker->submit(runPose); }
            catch (...) { obbDone.wait(); throw; }
            // Always await both before rethrowing: tasks capture this stack frame.
            std::exception_ptr workerError;
            try { obbDone.get(); } catch (...) { workerError = std::current_exception(); }
            try { poseDone.get(); } catch (...) { if (!workerError) workerError = std::current_exception(); }
            if (workerError) std::rethrow_exception(workerError);
        } else { runObb(); runPose(); }
        result->detections = detections; result->pose = bestPose; result->hasPose = bestScore >= 0;
        result->obbFrame = obbFrame; result->poseFrame = poseFrame;
        result->obbCalls = obbCalls; result->personCalls = personCalls; result->poseCalls = poseCalls;
        result->stepMs = milliseconds(started, PipelineClock::now());
        ++inferred;
        return result;
    };

    if (!args.headless) {
        cv::namedWindow("SOP C++ TensorRT", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::resizeWindow("SOP C++ TensorRT", args.displayWidth, args.displayHeight);
    }
    const auto started = PipelineClock::now();
    auto measuredStart = started;
    auto withinDuration = [&] { return args.duration <= 0 || milliseconds(started, PipelineClock::now()) < args.duration * 1000.; };
    auto read = [&](uint64_t id) -> FramePtr {
        auto frame = std::make_shared<PipelineFrame>(); frame->id = id;
        frame->captured = PipelineClock::now();
        if (!capture.read(frame->image)) return {};
        frame->decodeMs = milliseconds(frame->captured, PipelineClock::now());
        ++decoded; return frame;
    };
    if (threaded) {
        threads.emplace_back([&] {
            try {
                for (uint64_t id = 0; !stopped && withinDuration() && (!args.maxFrames || id < static_cast<uint64_t>(args.maxFrames)); ++id) {
                    if (args.realtime) {
                        const auto due = started + std::chrono::duration_cast<PipelineClock::duration>(std::chrono::duration<double>(id / fps));
                        std::this_thread::sleep_until(due);
                        if (stopped) break;
                    }
                    auto frame = read(id); if (!frame) break;
                    if (args.realtime) {
                        droppedJobs += inputs.pushLatest(frame);
                        droppedPreviews += previews.pushLatest(frame);
                    } else if (!inputs.push(frame)) break;
                }
                inputs.close(); previews.close();
            } catch (...) { fail(); }
        });
        threads.emplace_back([&] {
            try {
                checkCuda(cudaSetDevice(0), "inference cudaSetDevice");
                FramePtr frame;
                while (!stopped && inputs.pop(frame)) {
                    auto result = process(frame);
                    if (args.realtime) { std::lock_guard<std::mutex> lock(resultMutex); latestResult = result; }
                    else if (!results.push(result)) break;
                }
                results.close();
            } catch (...) { fail(); }
        });
    }
    uint64_t displayed = 0, measured = 0, skippedWarmup = 0, stalePose = 0;
    std::vector<double> gaps, latencies, poseAges, renderTimes;
    PipelineTime previousDisplay{};
    ResultPtr lastDisplayResult;
    cv::VideoWriter writer;
    while (!stopped && withinDuration()) {
        FramePtr frame; ResultPtr result;
        if (!threaded) {
            if (args.maxFrames && displayed >= static_cast<uint64_t>(args.maxFrames)) break;
            frame = read(displayed); if (!frame) break;
            result = process(frame);
        } else if (args.realtime) {
            if (!previews.popFor(frame, std::chrono::milliseconds(10))) {
                if (previews.finished()) break;
                if (!args.headless) { const int key = cv::waitKey(1); if (key == 27 || key == 'q' || key == 'Q') break; }
                continue;
            }
            { std::lock_guard<std::mutex> lock(resultMutex); result = latestResult; }
            if (result && result->frame->id <= frame->id) lastDisplayResult = result;
            result = lastDisplayResult;
        } else {
            if (!results.popFor(result, std::chrono::milliseconds(10))) {
                if (results.finished()) break;
                if (!args.headless) { const int key = cv::waitKey(1); if (key == 27 || key == 'q' || key == 'Q') break; }
                continue;
            }
            frame = result->frame;
        }
        const auto now = PipelineClock::now();
        const double gap = displayed ? milliseconds(previousDisplay, now) : 0;
        previousDisplay = now;
        const double latency = milliseconds(frame->captured, now);
        const double obbAge = result && result->obbFrame >= 0 ? (static_cast<int64_t>(frame->id) - result->obbFrame) * 1000. / fps : -1;
        const double poseAge = result && result->hasPose && result->poseFrame >= 0 ? (static_cast<int64_t>(frame->id) - result->poseFrame) * 1000. / fps : -1;
        const bool obbVisible = result && (!args.realtime || (obbAge >= 0 && obbAge <= args.maxResultAgeMs));
        const bool poseVisible = result && result->hasPose && (!args.realtime || (poseAge >= 0 && poseAge <= args.maxResultAgeMs));
        if (csv) {
            csv << frame->id << ',' << args.startSec * 1000. + frame->id * 1000. / fps << ',';
            if (result) csv << result->frame->id << ',' << result->obbFrame << ',' << result->poseFrame;
            else csv << "-1,-1,-1";
            csv << ',' << frame->decodeMs << ',' << (result ? result->stepMs : 0) << ',' << (result ? result->obbMs : 0)
                << ',' << (result ? result->personPoseMs : 0) << ',' << gap << ',' << latency << ',' << obbAge << ',' << poseAge
                << ',' << (result ? result->obbCalls : 0) << ',' << (result ? result->personCalls : 0)
                << ',' << (result ? result->poseCalls : 0) << ',' << (result ? resultFingerprint(*result) : 0) << '\n';
        }
        if (!args.headless || !args.record.empty()) {
            const auto renderStarted = PipelineClock::now();
            cv::Mat painted;
            double scaleX = 1., scaleY = 1.;
            if (args.previewFirst) {
                const double scale = std::min({1., static_cast<double>(args.displayWidth) / frame->image.cols,
                                               static_cast<double>(args.displayHeight) / frame->image.rows});
                if (scale < 1.) {
                    cv::resize(frame->image, painted, cv::Size(), scale, scale, cv::INTER_LINEAR);
                    scaleX = static_cast<double>(painted.cols) / frame->image.cols;
                    scaleY = static_cast<double>(painted.rows) / frame->image.rows;
                } else painted = frame->image.clone();
            } else painted = frame->image.clone();
            const std::vector<Detection> empty;
            std::vector<Detection> previewDetections;
            PoseResult previewPose;
            const auto* drawDetections = obbVisible ? &result->detections : &empty;
            const PoseResult* drawPose = poseVisible ? &result->pose : nullptr;
            if (args.previewFirst) {
                previewDetections = *drawDetections;
                for (auto& d : previewDetections) for (auto& p : d.points) { p.x *= scaleX; p.y *= scaleY; }
                drawDetections = &previewDetections;
                if (drawPose) {
                    previewPose = *drawPose;
                    for (auto& p : previewPose.keypoints) { p.x *= scaleX; p.y *= scaleY; }
                    previewPose.bbox.x *= scaleX; previewPose.bbox.y *= scaleY;
                    previewPose.bbox.width *= scaleX; previewPose.bbox.height *= scaleY;
                    drawPose = &previewPose;
                }
            }
            const double throughput = (displayed + 1) / std::max(milliseconds(started, now) / 1000., 1e-6);
            draw(painted, *drawDetections, drawPose,
                 throughput, result ? result->stepMs : 0);
            std::ostringstream label;
            label << args.mode << (args.realtime ? " LIVE" : " MATCHED") << " | frame " << frame->id
                  << " | pose age " << std::fixed << std::setprecision(0) << poseAge << "ms"
                  << (poseVisible ? "" : " (hidden)");
            cv::putText(painted, label.str(), cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, .65, cv::Scalar(255,255,255), 2);
            if (!args.record.empty()) {
                if (!writer.isOpened()) { writer.open(args.record, cv::VideoWriter::fourcc('m','p','4','v'), fps, painted.size()); if (!writer.isOpened()) throw std::runtime_error("Cannot open recording"); }
                writer.write(painted);
            }
            if (!args.headless) {
                const double scale = std::min({1., static_cast<double>(args.displayWidth) / painted.cols, static_cast<double>(args.displayHeight) / painted.rows});
                cv::Mat display;
                if (scale < 1.) cv::resize(painted, display, cv::Size(), scale, scale, cv::INTER_AREA); else display = painted;
                cv::imshow("SOP C++ TensorRT", display);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') break;
            }
            if (skippedWarmup >= static_cast<uint64_t>(args.warmupFrames))
                renderTimes.push_back(milliseconds(renderStarted, PipelineClock::now()));
        }
        ++displayed;
        if (skippedWarmup < static_cast<uint64_t>(args.warmupFrames)) { ++skippedWarmup; measuredStart = PipelineClock::now(); }
        else {
            ++measured; if (gap > 0) gaps.push_back(gap); latencies.push_back(latency);
            if (poseAge >= 0) poseAges.push_back(poseAge);
            if (result && result->hasPose && !poseVisible) ++stalePose;
        }
        if (displayed % 100 == 0) std::cout << "mode=" << args.mode << " displayed=" << displayed << " inferred=" << inferred << " obb_calls=" << (result ? result->obbCalls : 0) << " pose_calls=" << (result ? result->poseCalls : 0) << '\n';
    }
    const auto measuredEnd = PipelineClock::now();
    guard.finish();
    if (error) std::rethrow_exception(error);
    if (writer.isOpened()) writer.release();
    if (!args.headless) cv::destroyAllWindows();
    const double seconds = std::max(milliseconds(measuredStart, measuredEnd) / 1000., 1e-6);
    const double totalSeconds = std::max(milliseconds(started, measuredEnd) / 1000., 1e-6);
    std::cout << std::fixed << std::setprecision(3)
        << "status=complete mode=" << args.mode << " realtime=" << args.realtime
        << " decoded=" << decoded << " inferred=" << inferred << " displayed=" << displayed
        << " measured=" << measured << " elapsed_s=" << seconds << " processed_fps=" << measured / seconds
        << " actual_job_hz=" << inferred / totalSeconds << " obb_calls=" << obbCalls << " person_calls=" << personCalls
        << " pose_calls=" << poseCalls << " pose_hz=" << poseCalls / totalSeconds
        << " gap_p50_ms=" << percentile(gaps, .5) << " gap_p95_ms=" << percentile(gaps, .95)
        << " capture_to_display_p95_ms=" << percentile(latencies, .95)
        << " render_p95_ms=" << percentile(renderTimes, .95)
        << " pose_age_p95_ms=" << percentile(poseAges, .95) << " stale_pose_frames=" << stalePose
        << " dropped_jobs=" << droppedJobs << " dropped_previews=" << droppedPreviews << '\n';
    return 0;
}
#endif // !SOP_BATCH
