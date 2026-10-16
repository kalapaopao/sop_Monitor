// Capacity experiment, not a general multi-camera application.
// Embedding keeps model transforms/thresholds identical to the single-input runner.
#define SOP_INFER_EMBEDDED
#include "main.cpp"

namespace {
struct ProbeLane {
    std::mutex mutex;
    FramePtr preview, pending;
    ResultPtr result;
    double fps = 25.;
    std::atomic<uint64_t> decoded{0}, jobs{0}, dropped{0}, steadyJobs{0}, poseCalls{0};
    std::atomic<bool> eof{false};
};
int dualProbe(int argc, char** argv) {
    std::string second = "../test_vedio/07.mp4";
    std::vector<char*> filtered{argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "--source VIDEO --second-source VIDEO --duration SEC (>5) [--model-dir DIR] [--start-sec SEC] [--headless] [--metrics CSV] [--display-width PX] [--display-height PX]\n";
            return 0;
        }
        if (std::string(argv[i]) == "--second-source") {
            if (++i >= argc) throw std::runtime_error("Missing --second-source");
            second = argv[i];
        } else filtered.push_back(argv[i]);
    }
    // Reuse parsing/engine overrides; no pipeline workers are instantiated here.
    char modeKey[] = "--mode", modeValue[] = "serial";
    filtered.push_back(modeKey); filtered.push_back(modeValue);
    Args args = parseArgs(static_cast<int>(filtered.size()), filtered.data());
    if (args.record.size() || args.previewFirst || args.realtime || args.maxFrames || args.warmupFrames ||
        args.obbStride != 1 || args.personStride != 1 || args.poseStride != 1)
        throw std::runtime_error("Dual probe uses realtime full-model jobs; record/stride/frame-limit options are unsupported");
    if (args.duration <= 5 || !std::isfinite(args.duration))
        throw std::runtime_error("Dual probe requires --duration greater than 5 seconds (first 5s warmup)");
    if (!args.headless) prepareDesktopDisplay();
    checkCuda(cudaSetDevice(0), "dual cudaSetDevice");
    TrtModel obb(args.yoloEngine), person(args.yoloxEngine), poseModel(args.rtmwEngine);
    const std::array<std::string,2> sources{{args.source, second}};
    std::array<cv::VideoCapture,2> captures;
    std::array<ProbeLane,2> lanes;
    for (int i = 0; i < 2; ++i) {
        if (!captures[i].open(sources[i])) throw std::runtime_error("Cannot open " + sources[i]);
        const double rate = captures[i].get(cv::CAP_PROP_FPS);
        if (rate > 0 && std::isfinite(rate)) lanes[i].fps = rate;
        if (args.startSec && !captures[i].set(cv::CAP_PROP_POS_MSEC, args.startSec * 1000.))
            throw std::runtime_error("Cannot seek " + sources[i]);
        std::cout << "lane=" << i << " source=" << sources[i] << " width=" << captures[i].get(cv::CAP_PROP_FRAME_WIDTH)
                  << " height=" << captures[i].get(cv::CAP_PROP_FRAME_HEIGHT) << " fps=" << lanes[i].fps << '\n';
    }
    std::ofstream csv;
    if (!args.metrics.empty()) {
        csv.open(args.metrics);
        if (!csv) throw std::runtime_error("Cannot open metrics");
        csv << "lane,elapsed_ms,frame,result_frame,pose_age_ms,jobs,pose_calls,visible\n";
    }
    if (!args.headless) {
        cv::namedWindow("SOP two stations | shared serial models", cv::WINDOW_NORMAL);
        cv::resizeWindow("SOP two stations | shared serial models", args.displayWidth, args.displayHeight);
    }
    const auto start = PipelineClock::now();
    auto elapsed = [&] { return milliseconds(start, PipelineClock::now()); };
    std::atomic<bool> stop{false};
    std::exception_ptr error;
    std::mutex errorMutex;
    std::vector<std::thread> threads;
    auto fail = [&] { std::lock_guard<std::mutex> lock(errorMutex); if (!error) error = std::current_exception(); stop = true; };
    struct Guard {
        std::atomic<bool>& stop;
        std::vector<std::thread>& threads;
        void join() { stop = true; for (auto& t : threads) if (t.joinable()) t.join(); }
        ~Guard() { join(); }
    } guard{stop,threads};
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&,i] {
            try {
                for (uint64_t id = 0; !stop && elapsed() < args.duration * 1000.; ++id) {
                    const auto due = start + std::chrono::duration_cast<PipelineClock::duration>(std::chrono::duration<double>(id / lanes[i].fps));
                    std::this_thread::sleep_until(due);
                    if (stop) break;
                    auto f = std::make_shared<PipelineFrame>(); f->id = id; f->captured = PipelineClock::now();
                    if (!captures[i].read(f->image)) break;
                    f->decodeMs = milliseconds(f->captured, PipelineClock::now());
                    ++lanes[i].decoded;
                    std::lock_guard<std::mutex> lock(lanes[i].mutex);
                    if (lanes[i].pending) ++lanes[i].dropped;
                    lanes[i].pending = f; lanes[i].preview = f;
                }
                lanes[i].eof = true;
            } catch (...) { fail(); }
        });
    }
    threads.emplace_back([&] {
        try {
            checkCuda(cudaSetDevice(0), "dual worker cudaSetDevice");
            int next = 0;
            while (!stop && elapsed() < args.duration * 1000.) {
                FramePtr f; int selected = -1;
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = (next + offset) % 2;
                    std::lock_guard<std::mutex> lock(lanes[i].mutex);
                    if (lanes[i].pending) { f = lanes[i].pending; lanes[i].pending.reset(); selected = i; break; }
                }
                if (!f) {
                    if (lanes[0].eof && lanes[1].eof) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue;
                }
                next = 1 - selected; // Fair round-robin, one completed job per lane turn.
                auto result = std::make_shared<PipelineResult>(); result->frame = f;
                const auto t = PipelineClock::now();
                result->detections = yoloDetections(obb, f->image, .25f);
                auto boxes = yoloXPeople(person, f->image);
                float best = -1.; uint64_t crops = 0;
                for (const auto& box : boxes) {
                    auto pose = rtmwPose(poseModel, f->image, box); ++crops;
                    const float score = std::accumulate(pose.scores.begin()+5, pose.scores.begin()+11, 0.f) / 6.f;
                    if (score > best) { best = score; result->pose = pose; result->hasPose = true; }
                }
                result->obbFrame = static_cast<int64_t>(f->id);
                result->poseFrame = result->hasPose ? static_cast<int64_t>(f->id) : -1;
                result->stepMs = milliseconds(t, PipelineClock::now());
                ++lanes[selected].jobs; lanes[selected].poseCalls += crops;
                const double completedMs = elapsed();
                if (completedMs >= 5000. && completedMs < args.duration * 1000.) ++lanes[selected].steadyJobs;
                std::lock_guard<std::mutex> lock(lanes[selected].mutex);
                lanes[selected].result = result;
            }
        } catch (...) { fail(); }
    });
    std::array<uint64_t,2> displayed{{0,0}}, measured{{0,0}}, hidden{{0,0}};
    std::array<int64_t,2> lastFrame{{-1,-1}};
    std::array<ResultPtr,2> eligible;
    std::array<std::vector<double>,2> ages;
    const int tileW = args.displayWidth / 2, tileH = args.displayHeight;
    while (!stop && elapsed() < args.duration * 1000.) {
        std::array<FramePtr,2> frames;
        std::array<ResultPtr,2> snapshots;
        bool changed = false;
        for (int i = 0; i < 2; ++i) {
            std::lock_guard<std::mutex> lock(lanes[i].mutex);
            frames[i] = lanes[i].preview; snapshots[i] = lanes[i].result;
            changed = changed || (frames[i] && static_cast<int64_t>(frames[i]->id) != lastFrame[i]);
        }
        if (!changed) {
            if (!args.headless) { const int key = cv::waitKey(1); if (key == 27 || key == 'q' || key == 'Q') break; }
            else std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (lanes[0].eof && lanes[1].eof) break;
            continue;
        }
        cv::Mat mosaic;
        if (!args.headless) mosaic = cv::Mat::zeros(tileH, tileW * 2, CV_8UC3);
        for (int i = 0; i < 2; ++i) {
            FramePtr f = frames[i]; ResultPtr r = snapshots[i];
            if (!f) continue;
            if (r && r->frame->id <= f->id) eligible[i] = r;
            r = eligible[i];
            const double age = r && r->hasPose ? (f->id - r->frame->id) * 1000. / lanes[i].fps : -1.;
            const double obbAge = r ? (f->id - r->frame->id) * 1000. / lanes[i].fps : -1.;
            const bool visible = age >= 0 && age <= args.maxResultAgeMs;
            if (static_cast<int64_t>(f->id) != lastFrame[i]) {
                lastFrame[i] = static_cast<int64_t>(f->id); ++displayed[i];
                if (elapsed() >= 5000.) { ++measured[i]; if (!visible) ++hidden[i]; if (age >= 0) ages[i].push_back(age); }
                if (csv) csv << i << ',' << elapsed() << ',' << f->id << ',' << (r ? static_cast<int64_t>(r->frame->id) : -1)
                             << ',' << age << ',' << lanes[i].jobs << ',' << lanes[i].poseCalls << ',' << visible << '\n';
            }
            if (!args.headless) {
                const double scale = std::min(static_cast<double>(tileW) / f->image.cols, static_cast<double>(tileH) / f->image.rows);
                cv::Mat image; cv::resize(f->image, image, cv::Size(), scale, scale, cv::INTER_LINEAR);
                const double sx = static_cast<double>(image.cols) / f->image.cols, sy = static_cast<double>(image.rows) / f->image.rows;
                std::vector<Detection> ds;
                PoseResult pose;
                if (r && obbAge <= args.maxResultAgeMs) {
                    ds = r->detections;
                    for (auto& d : ds) for (auto& p : d.points) { p.x *= sx; p.y *= sy; }
                }
                if (visible) {
                    pose = r->pose;
                    for (auto& p : pose.keypoints) { p.x *= sx; p.y *= sy; }
                    pose.bbox.x *= sx; pose.bbox.y *= sy; pose.bbox.width *= sx; pose.bbox.height *= sy;
                }
                draw(image, ds, visible ? &pose : nullptr, displayed[i] / std::max(elapsed()/1000.,1e-6), r ? r->stepMs : 0.);
                image.copyTo(mosaic(cv::Rect(i*tileW,0,image.cols,image.rows)));
                std::ostringstream text; text << "station " << i+1 << " " << sources[i] << " | pose age " << age << "ms";
                cv::putText(mosaic,text.str(),cv::Point(i*tileW+8,std::min(tileH-15,image.rows+25)),cv::FONT_HERSHEY_SIMPLEX,.43,cv::Scalar(255,255,255),1);
            }
        }
        if (!args.headless) {
            cv::imshow("SOP two stations | shared serial models", mosaic);
            const int key = cv::waitKey(1); if (key == 27 || key == 'q' || key == 'Q') break;
        } else std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (lanes[0].eof && lanes[1].eof) break;
    }
    const double measurementEnd = elapsed();
    guard.join();
    if (error) std::rethrow_exception(error);
    if (!args.headless) cv::destroyAllWindows();
    for (int i = 0; i < 2; ++i)
        std::cout << std::fixed << std::setprecision(3) << "status=complete lane=" << i << " shared_models=1 serial=1"
                  << " duration_s=" << measurementEnd/1000. << " decoded=" << lanes[i].decoded << " jobs=" << lanes[i].jobs
                  << " steady_job_hz=" << lanes[i].steadyJobs / std::max((measurementEnd-5000.)/1000.,1e-6)
                  << " pose_calls=" << lanes[i].poseCalls << " displayed=" << displayed[i] << " dropped_jobs=" << lanes[i].dropped
                  << " pose_age_p95_ms=" << percentile(ages[i],.95)
                  << " hidden_fraction=" << static_cast<double>(hidden[i])/std::max<uint64_t>(measured[i],1) << '\n';
    return 0;
}
}
int main(int argc, char** argv) {
    try { return dualProbe(argc,argv); }
    catch (const std::exception& e) { std::cerr << "status=failed error=" << e.what() << '\n'; return 1; }
}
