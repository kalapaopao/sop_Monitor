#define SOP_BATCH
#define SOP_STAGE2
#define SOP_ACCEL
#define SOP_JOINT_DETECTOR
#include "batch_kernels.hpp"
// Two/four-input capacity experiment, not a general multi-camera application.
// Embedding keeps model transforms/thresholds identical to the single-input runner.
#define SOP_INFER_EMBEDDED
#include "main.cpp"
#ifdef SOP_ACCEL
#include "hardware_capture.hpp"
#endif
#ifdef SOP_JOINT_DETECTOR
extern "C" int sopRegisterSiLU();
#endif

namespace {
#ifdef SOP_JOINT_DETECTOR
#include "joint_detector.hpp"
#include "batch_decode.hpp"
#include "batch_runner.hpp"
#endif
struct ProbeLane {
    std::mutex mutex;
    FramePtr preview, pending;
    ResultPtr result;
    double fps = 25.;
    std::atomic<uint64_t> decoded{0}, jobs{0}, dropped{0}, steadyJobs{0}, poseCalls{0};
    std::atomic<bool> eof{false};
};
int dualProbe(int argc, char** argv) {
    int detectorBatch=1,poseBatch=1,detectorWidth=640,detectorHeight=640;
    double batchWaitMs=5;
    std::string sampleDir,validationDir,batchMetrics;
    std::string second = "../test_vedio/07.mp4", third, fourth;
    bool singleSource = false;
    std::string auditDir;
    int workerCount = 1;
#ifdef SOP_STAGE2
    std::string posePreprocessing="cuda", sourceLayout="bgr", reusableIO="on";
#endif
#ifdef SOP_ACCEL
    workerCount = 1;
    std::string decoder = "gst-h265", preprocessing = "cuda";
#endif
    std::vector<char*> filtered{argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "--source VIDEO [--single-source | --second-source VIDEO [--third-source VIDEO --fourth-source VIDEO]] --duration SEC (>5) [--model-dir DIR] [--yolo-engine PLAN] [--start-sec SEC] [--headless] [--metrics CSV] [--audit-dir EXISTING_DIR] [--display-width PX] [--display-height PX]\n";
#ifdef SOP_ACCEL
            std::cout << "--workers 1|2|4 --decoder opencv|gst-h265 --preprocess cpu|cuda (accelerated target only)\n";
#ifdef SOP_STAGE2
            std::cout << "--pose-preprocess cpu|cuda --source-layout bgr|i420 --reuse-io on|off (stage2 experiments)\n";
#endif
#endif
            std::cout<<"--det-batch 1|2|4 --pose-batch 1|2|4 --det-height 384|640 --det-width 640 --batch-wait-ms 0..20 --batch-metrics CSV --sample-dir DIR --validation-dir EXISTING_DIR\n";
            return 0;
        }
        const std::string key = argv[i];
        if(key=="--det-batch" || key=="--pose-batch" || key=="--det-height" || key=="--det-width" ||
           key=="--batch-wait-ms" || key=="--sample-dir" || key=="--validation-dir" || key=="--batch-metrics") {
            if(++i>=argc)throw std::runtime_error("Missing "+key);
            if(key=="--det-batch")detectorBatch=std::stoi(argv[i]);
            else if(key=="--pose-batch")poseBatch=std::stoi(argv[i]);
            else if(key=="--det-height")detectorHeight=std::stoi(argv[i]);
            else if(key=="--det-width")detectorWidth=std::stoi(argv[i]);
            else if(key=="--batch-wait-ms")batchWaitMs=std::stod(argv[i]);
            else if(key=="--sample-dir")sampleDir=argv[i];
            else if(key=="--validation-dir")validationDir=argv[i];
            else batchMetrics=argv[i];
            continue;
        }
#ifdef SOP_STAGE2
        if(key=="--pose-preprocess" || key=="--source-layout" || key=="--reuse-io") {
            if(++i>=argc) throw std::runtime_error("Missing "+key);
            if(key=="--pose-preprocess") posePreprocessing=argv[i];
            else if(key=="--source-layout") sourceLayout=argv[i];
            else reusableIO=argv[i];
            continue;
        }
#endif
#ifdef SOP_ACCEL
        if (key=="--workers" || key=="--decoder" || key=="--preprocess") {
            if (++i>=argc) throw std::runtime_error("Missing "+key);
            if (key=="--workers") workerCount=std::stoi(argv[i]);
            else if (key=="--decoder") decoder=argv[i];
            else preprocessing=argv[i];
            continue;
        }
#endif
        if (key == "--audit-dir") {
            if (++i >= argc) throw std::runtime_error("Missing --audit-dir");
            auditDir = argv[i]; continue;
        }
        if (key == "--single-source") { singleSource = true; continue; }
        if (key == "--second-source" || key == "--third-source" || key == "--fourth-source") {
            if (++i >= argc) throw std::runtime_error("Missing " + key);
            if (key == "--second-source") second = argv[i];
            else if (key == "--third-source") third = argv[i];
            else fourth = argv[i];
        } else filtered.push_back(argv[i]);
    }
    if (third.empty() != fourth.empty())
        throw std::runtime_error("--third-source and --fourth-source must be supplied together");
    if (singleSource && !third.empty()) throw std::runtime_error("--single-source cannot have third/fourth inputs");
    const int laneCount = singleSource ? 1 : (third.empty() ? 2 : 4);
#ifdef SOP_ACCEL
    if ((workerCount!=1 && workerCount!=2 && workerCount!=4) || workerCount>laneCount)
        throw std::runtime_error("workers must be 1,2,4 and not exceed the input count");
    if (preprocessing!="cpu" && preprocessing!="cuda") throw std::runtime_error("preprocess must be cpu or cuda");
#ifdef SOP_STAGE2
    if((posePreprocessing!="cpu" && posePreprocessing!="cuda") || (sourceLayout!="bgr" && sourceLayout!="i420") ||
       (reusableIO!="on" && reusableIO!="off")) throw std::runtime_error("Invalid stage2 option");
    if(sourceLayout=="i420" && decoder!="gst-h265") throw std::runtime_error("I420 requires hardware H.265 capture");
    if(sourceLayout=="i420") throw std::runtime_error("I420 GPU conversion failed pose parity; use stage2_test --i420 for diagnostics");
#endif
#endif
    if(workerCount!=1 || preprocessing!="cuda" || posePreprocessing!="cuda" || reusableIO!="on")
        throw std::runtime_error("Batch probe requires one worker, CUDA preprocessing and reused I/O");
    if((detectorBatch!=1 && detectorBatch!=2 && detectorBatch!=4) || (poseBatch!=1 && poseBatch!=2 && poseBatch!=4) ||
       detectorWidth!=640 || (detectorHeight!=640 && detectorHeight!=384) || !std::isfinite(batchWaitMs) || batchWaitMs<0 || batchWaitMs>20)
        throw std::runtime_error("Invalid batch experiment configuration");
    if(sampleDir.empty()!=validationDir.empty())throw std::runtime_error("Both validation options are required");
    if(!auditDir.empty())throw std::runtime_error("Use --sample-dir/--validation-dir for batch audits");
    const int gridCols = laneCount == 1 ? 1 : 2;
    const int gridRows = laneCount == 4 ? 2 : 1;
    std::string windowTitle = "SOP " + std::to_string(laneCount) + " stations | shared serial models";
#ifdef SOP_JOINT_DETECTOR
    windowTitle += " | BATCH OBB + RTMW";
#endif
#ifdef SOP_ACCEL
    windowTitle = "SOP accelerated | " + std::to_string(laneCount) + " lanes | " + std::to_string(workerCount) + " workers";
#ifdef SOP_STAGE2
    windowTitle="SOP stage2 | "+std::to_string(laneCount)+" lanes | "+posePreprocessing+" pose | "+sourceLayout;
#endif
#endif
    // Reuse parsing/engine overrides; no pipeline workers are instantiated here.
    char modeKey[] = "--mode", modeValue[] = "serial";
    filtered.push_back(modeKey); filtered.push_back(modeValue);
    Args args = parseArgs(static_cast<int>(filtered.size()), filtered.data());
#ifdef SOP_ACCEL
    if(decoder=="gst-h265" && args.startSec!=0)
        throw std::runtime_error("Hardware file seeking is unsupported by this backend; use --decoder opencv for --start-sec");
#endif
#ifdef SOP_JOINT_DETECTOR
    const bool explicitEngine = std::any_of(filtered.begin(),filtered.end(),[](const char* s) { return std::string(s)=="--yolo-engine"; });
    if(!explicitEngine)args.yoloEngine=args.modelDir+"/batch_20261020/obb_640x"+std::to_string(detectorHeight)+"_b1_4.plan";
    const bool explicitPose=std::any_of(filtered.begin(),filtered.end(),[](const char* s){return std::string(s)=="--rtmw-engine";});
    if(!explicitPose)args.rtmwEngine=poseBatch==1?args.modelDir+"/../build/rtmw.plan":
        args.modelDir+"/batch_20261020/rtmw_b"+std::to_string(poseBatch)+".plan";
#endif
    if (args.record.size() || args.previewFirst || args.realtime || args.maxFrames || args.warmupFrames ||
        args.obbStride != 1 || args.personStride != 1 || args.poseStride != 1)
        throw std::runtime_error("Dual probe uses realtime full-model jobs; record/stride/frame-limit options are unsupported");
    if (args.duration <= 5 || !std::isfinite(args.duration))
        throw std::runtime_error("Dual probe requires --duration greater than 5 seconds (first 5s warmup)");
    if (!args.headless && sampleDir.empty()) prepareDesktopDisplay();
    checkCuda(cudaSetDevice(0), "dual cudaSetDevice");
#ifdef SOP_JOINT_DETECTOR
    if (sopRegisterSiLU() != 1) throw std::runtime_error("SiLU plugin registration failed");
#ifdef SOP_ACCEL
    auto detectorRoot=std::make_shared<TrtModel>(args.yoloEngine);
    auto poseRoot=std::make_shared<TrtModel>(args.rtmwEngine);
    std::vector<std::shared_ptr<TrtModel>> detectors{detectorRoot}, poses{poseRoot};
    for (int i=1;i<workerCount;++i) {
        detectors.push_back(std::make_shared<TrtModel>(detectorRoot));
        poses.push_back(std::make_shared<TrtModel>(poseRoot));
    }
    for (auto& detector:detectors) detector->gpuPreprocess=preprocessing=="cuda";
#ifdef SOP_STAGE2
    for(auto& detector:detectors) detector->reuseIO=reusableIO=="on";
    for(auto& pose:poses) pose->reuseIO=reusableIO=="on";
    std::cout<<"pose_preprocess="<<posePreprocessing<<" source_layout="<<sourceLayout<<" reuse_io="<<reusableIO<<'\n';
#endif
    std::cout << "workers=" << workerCount << " decoder=" << decoder << " preprocessing=" << preprocessing
              << " engines=2 contexts=" << workerCount*2 << " per_lane_inflight=1\n";
#else
    TrtModel obb(args.yoloEngine), poseModel(args.rtmwEngine);
#endif
    std::cout << "detector=joint_obb models=2 person_threshold=0.25 nms=rotated_classwise_0.55\n";
#else
    TrtModel obb(args.yoloEngine), person(args.yoloxEngine), poseModel(args.rtmwEngine);
#endif
    std::cout<<"detector_width="<<detectorWidth<<" detector_height="<<detectorHeight
             <<" detector_batch="<<detectorBatch<<" pose_batch="<<poseBatch<<" batch_wait_ms="<<batchWaitMs<<'\n';
    if(!sampleDir.empty()) {
        validateBatchSamples(*detectorRoot,*poseRoot,sampleDir,validationDir,detectorWidth,detectorHeight,detectorBatch,poseBatch);
        return 0;
    }
    std::ofstream groupCsv;
    if(!batchMetrics.empty()) {
        groupCsv.open(batchMetrics);if(!groupCsv)throw std::runtime_error("Cannot open batch metrics");
        groupCsv<<"elapsed_ms,group_size,source_ms,detector_ms,pose_ms,total_ms,detector_calls,pose_calls\n";
    }
    std::array<uint64_t,5> detectorSizes{},poseSizes{};
    const std::array<std::string,4> sources{{args.source, second, third, fourth}};
#ifdef SOP_ACCEL
    std::array<AcceleratedCapture,4> captures;
#else
    std::array<cv::VideoCapture,4> captures;
#endif
    std::array<ProbeLane,4> lanes;
    for (int i = 0; i < laneCount; ++i) {
#ifdef SOP_ACCEL
        openAcceleratedCapture(captures[i],sources[i],decoder);
#ifdef SOP_STAGE2
        captures[i].keepNative=sourceLayout=="i420";
#endif
        std::cout << "capture_backend=" << captures[i].getBackendName() << '\n';
#else
        if (!captures[i].open(sources[i])) throw std::runtime_error("Cannot open " + sources[i]);
#endif
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
        csv << "lane,elapsed_ms,frame,result_frame,pose_age_ms,jobs,pose_calls,visible,detector_ms,pose_ms,person_count,result_hash\n";
    }
    if (!args.headless) {
        cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowTitle, args.displayWidth, args.displayHeight);
    }
    auto allEof = [&] {
        return std::all_of(lanes.begin(), lanes.begin() + laneCount,
                           [](const ProbeLane& lane) { return lane.eof.load(); });
    };
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
    for (int i = 0; i < laneCount; ++i) {
        threads.emplace_back([&,i] {
            try {
                for (uint64_t id = 0; !stop && elapsed() < args.duration * 1000.; ++id) {
                    const auto due = start + std::chrono::duration_cast<PipelineClock::duration>(std::chrono::duration<double>(id / lanes[i].fps));
                    std::this_thread::sleep_until(due);
                    if (stop) break;
                    auto f = std::make_shared<PipelineFrame>(); f->id = id; f->captured = PipelineClock::now();
                    if (!captures[i].read(f->image)) break;
#ifdef SOP_STAGE2
                    if(sourceLayout=="i420") {
                        f->nativeI420=captures[i].nativeI420();
                        const auto color=captures[i].sourceColor(); f->colorMatrix=color.matrix; f->fullRange=color.fullRange;
                        if(f->nativeI420.empty()) throw std::runtime_error("Native source unavailable");
                    }
#endif
                    f->decodeMs = milliseconds(f->captured, PipelineClock::now());
#ifdef SOP_PROFILE
                    f->profileReadMs=captures[i].profileReadMs; f->profileColorMs=captures[i].profileColorMs;
                    sop_profile::standalone(i,f->id,elapsed(),"capture",{
                        {"capture","read_wait_download",f->profileReadMs},
                        {"capture","color_conversion",f->profileColorMs},
                        {"capture","decode_total",f->decodeMs}});
#endif
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
            checkCuda(cudaSetDevice(0),"batch worker device");
            int next=0;const size_t groupLimit=std::min(laneCount,std::max(detectorBatch,poseBatch));
            while(!stop && elapsed()<args.duration*1000.) {
                std::vector<FramePtr> frames;std::vector<int> owners;std::array<bool,4> taken{};
                const auto deadline=PipelineClock::now()+std::chrono::duration<double,std::milli>(batchWaitMs);
                do {
                    for(int offset=0;offset<laneCount && frames.size()<groupLimit;++offset) {
                        const int i=(next+offset)%laneCount;if(taken[i])continue;
                        std::lock_guard<std::mutex> lock(lanes[i].mutex);
                        if(lanes[i].pending) {
                            frames.push_back(lanes[i].pending);lanes[i].pending.reset();owners.push_back(i);taken[i]=true;
                        }
                    }
                    if(frames.size()==groupLimit || frames.empty() || PipelineClock::now()>=deadline)break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } while(!stop);
                if(frames.empty()) {
                    if(allEof())break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;
                }
                next=(owners.back()+1)%laneCount;
                const auto run=inferGroup(*detectorRoot,*poseRoot,frames,detectorWidth,detectorHeight,detectorBatch,poseBatch);
                for(int n:run.detectorSizes)++detectorSizes[n];
                for(int n:run.poseSizes)++poseSizes[n];
                const double completedMs=elapsed();
                if(groupCsv)groupCsv<<completedMs<<','<<frames.size()<<','<<run.sourceMs<<','<<run.detectorMs<<','
                    <<run.poseMs<<','<<run.totalMs<<','<<run.detectorSizes.size()<<','<<run.poseSizes.size()<<'\n';
                for(size_t k=0;k<owners.size();++k) {
                    const int selected=owners[k];const auto& result=run.results[k];
                    ++lanes[selected].jobs;lanes[selected].poseCalls+=result->poseCalls;
                    if(completedMs>=5000. && completedMs<args.duration*1000.)++lanes[selected].steadyJobs;
                    std::lock_guard<std::mutex> lock(lanes[selected].mutex);lanes[selected].result=result;
                }
            }
        } catch(...) { fail(); }
    });
    std::array<uint64_t,4> displayed{}, measured{}, hidden{};
    std::array<int64_t,4> lastFrame; lastFrame.fill(-1);
    std::array<ResultPtr,4> eligible;
    std::array<std::vector<double>,4> ages;
    const int tileW = args.displayWidth / gridCols, tileH = args.displayHeight / gridRows;
    cv::Mat mosaic;
    if (!args.headless) mosaic = cv::Mat::zeros(tileH * gridRows, tileW * gridCols, CV_8UC3);
    double displayFps = 1.;
    for (int i = 0; i < laneCount; ++i) displayFps = std::max(displayFps, lanes[i].fps);
    const auto displayPeriod = std::chrono::duration_cast<PipelineClock::duration>(std::chrono::duration<double>(1. / displayFps));
    auto nextDisplay = start;
    uint64_t paints = 0;
    while (!stop && elapsed() < args.duration * 1000.) {
        if (!args.headless) {
            // Present one batch at the source cadence rather than presenting
            // separately whenever any of the independent decoders publishes.
            std::this_thread::sleep_until(nextDisplay);
            nextDisplay = PipelineClock::now() + displayPeriod;
            if (stop || elapsed() >= args.duration * 1000.) break;
        }
        std::array<FramePtr,4> frames;
        std::array<ResultPtr,4> snapshots;
        bool changed = false;
        for (int i = 0; i < laneCount; ++i) {
            std::lock_guard<std::mutex> lock(lanes[i].mutex);
            frames[i] = lanes[i].preview; snapshots[i] = lanes[i].result;
            changed = changed || (frames[i] && static_cast<int64_t>(frames[i]->id) != lastFrame[i]);
        }
        if (!changed) {
            if (!args.headless) { const int key = cv::waitKey(1); if (key == 27 || key == 'q' || key == 'Q') break; }
            else std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (allEof()) break;
            continue;
        }
#ifdef SOP_PROFILE
        const auto composeStarted=PipelineClock::now();
#endif
        for (int i = 0; i < laneCount; ++i) {
            FramePtr f = frames[i]; ResultPtr r = snapshots[i];
            if (!f) continue;
            if (r && r->frame->id <= f->id) eligible[i] = r;
            r = eligible[i];
            const double age = r && r->hasPose ? (f->id - r->frame->id) * 1000. / lanes[i].fps : -1.;
            const double obbAge = r ? (f->id - r->frame->id) * 1000. / lanes[i].fps : -1.;
            const bool visible = age >= 0 && age <= args.maxResultAgeMs;
            const bool newFrame = static_cast<int64_t>(f->id) != lastFrame[i];
            if (newFrame) {
                lastFrame[i] = static_cast<int64_t>(f->id); ++displayed[i];
                if (elapsed() >= 5000.) { ++measured[i]; if (!visible) ++hidden[i]; if (age >= 0) ages[i].push_back(age); }
                if (csv) csv << i << ',' << elapsed() << ',' << f->id << ',' << (r ? static_cast<int64_t>(r->frame->id) : -1)
                             << ',' << age << ',' << lanes[i].jobs << ',' << lanes[i].poseCalls << ',' << visible
                             << ',' << (r ? r->obbMs : 0) << ',' << (r ? r->personPoseMs : 0)
                             << ',' << (r ? r->personCalls : 0) << ',' << (r ? resultFingerprint(*r) : 0) << '\n';
            }
            if (!args.headless && newFrame) {
                const int originX = (i % gridCols) * tileW, originY = (i / gridCols) * tileH;
                const double scale = std::min(static_cast<double>(tileW) / f->image.cols, static_cast<double>(tileH - 24) / f->image.rows);
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
                mosaic(cv::Rect(originX,originY,tileW,tileH)).setTo(cv::Scalar());
                image.copyTo(mosaic(cv::Rect(originX,originY,image.cols,image.rows)));
                const auto file = sources[i].substr(sources[i].find_last_of("/\\") + 1);
                std::ostringstream text;
                text << i+1 << " " << file << " | jobs " << std::fixed << std::setprecision(1)
                     << lanes[i].jobs / std::max(elapsed()/1000.,1e-6) << "Hz | pose age "
                     << std::setprecision(0) << age << "ms" << (visible ? "" : " HIDDEN");
                cv::putText(mosaic,text.str(),cv::Point(originX+8,originY+tileH-8),cv::FONT_HERSHEY_SIMPLEX,.43,cv::Scalar(255,255,255),1);
            }
        }
        if (!args.headless) {
#ifdef SOP_PROFILE
            const auto presentStarted=PipelineClock::now();
#endif
            cv::imshow(windowTitle, mosaic);
            ++paints;
            const int key = cv::waitKey(1);
#ifdef SOP_PROFILE
            sop_profile::standalone(-1,paints,elapsed(),"display",{
                {"display","compose",milliseconds(composeStarted,presentStarted)},
                {"display","present_wait",milliseconds(presentStarted,PipelineClock::now())}});
#endif
            if (key == 27 || key == 'q' || key == 'Q') break;
        } else std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (allEof()) break;
    }
    const double measurementEnd = elapsed();
    guard.join();
    if (error) std::rethrow_exception(error);
#ifdef SOP_STAGE2
    for(int i=0;i<workerCount;++i) { detectors[i]->reportIO("detector"); poses[i]->reportIO("pose"); }
#endif
    if (!args.headless) cv::destroyAllWindows();
    for (int i = 0; i < laneCount; ++i)
        std::cout << std::fixed << std::setprecision(3) << "status=complete lane=" << i << " shared_models=1 serial=" << (workerCount==1) << " workers=" << workerCount << " lanes=" << laneCount
                  << " duration_s=" << measurementEnd/1000. << " decoded=" << lanes[i].decoded << " jobs=" << lanes[i].jobs
                  << " steady_job_hz=" << lanes[i].steadyJobs / std::max((measurementEnd-5000.)/1000.,1e-6)
                  << " pose_calls=" << lanes[i].poseCalls << " displayed=" << displayed[i] << " dropped_jobs=" << lanes[i].dropped
                  << " mosaic_presents=" << paints
                  << " pose_age_p95_ms=" << percentile(ages[i],.95)
                  << " hidden_fraction=" << static_cast<double>(hidden[i])/std::max<uint64_t>(measured[i],1) << '\n';
    for(int n=1;n<=4;++n)std::cout<<"actual_batch="<<n<<" detector_calls="<<detectorSizes[n]<<" pose_calls="<<poseSizes[n]<<'\n';
    return 0;
}
}
int main(int argc, char** argv) {
    try { return dualProbe(argc,argv); }
    catch (const std::exception& e) { std::cerr << "status=failed error=" << e.what() << '\n'; return 1; }
}
