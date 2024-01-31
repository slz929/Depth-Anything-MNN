#include <stdio.h>
#include <chrono>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <cv/cv.hpp>

#include <opencv2/opencv.hpp>

#include<iostream>

using namespace MNN;
using namespace MNN::Express;


int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./dam dam.mnn input.jpg [forwardType] [precision] [thread]\n");
        return 0;
    }
    int thread = 4;
    int precision = 0;
    int forwardType = MNN_FORWARD_CPU;
    if (argc >= 4) {
        forwardType = atoi(argv[3]);
    }
    if (argc >= 5) {
        precision = atoi(argv[4]);
    }
    if (argc >= 6) {
        thread = atoi(argv[5]);
    }
    float mask_threshold = 0;
    MNN::ScheduleConfig sConfig;
    sConfig.type = static_cast<MNNForwardType>(forwardType);
    sConfig.numThread = thread;
    BackendConfig bConfig;
    bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
    sConfig.backendConfig = &bConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(sConfig));
    if(rtmgr == nullptr) {
        MNN_ERROR("Empty RuntimeManger\n");
        return 0;
    }
    // rtmgr->setCache(".cachefile");
    std::shared_ptr<Module> embed(Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1], rtmgr));
    auto image = MNN::CV::imread(argv[2]);

    // 1. preprocess
    auto dims = image->getInfo()->dim;
    int origin_h = dims[0];
    int origin_w = dims[1];
    int length = 518;
    int new_h, new_w;
    if (origin_h > origin_w) {
        new_w = round(origin_w * (float)length / origin_h);
        new_h = length;
    } else {
        new_h = round(origin_h * (float)length / origin_w);
        new_w = length;
    }
    float scale_w = (float)new_w / origin_w;
    float scale_h = (float)new_h / origin_h;
    auto input_var = MNN::CV::resize(image, MNN::CV::Size(new_w, new_h), 0, 0, MNN::CV::INTER_CUBIC, -1, {123.675, 116.28, 103.53}, {1/58.395, 1/57.12, 1/57.375});
    std::vector<int> padvals { 0, length - new_h, 0, length - new_w, 0, 0 };
    auto pads = _Const(static_cast<void*>(padvals.data()), {3, 2}, NCHW, halide_type_of<int>());
    input_var = _Pad(input_var, pads, CONSTANT);
    input_var = _Unsqueeze(input_var, {0});
    // 2. forward
    for(int i =0; i< input_var->getInfo()->dim.size(); i++)
        std::cout<< input_var->getInfo()->dim[i]<<" ";
    std::cout<<"\n";

    input_var = _Convert(input_var, NC4HW4);

    for(int i =0; i< input_var->getInfo()->dim.size(); i++)
        std::cout<< input_var->getInfo()->dim[i]<<" ";
    std::cout<<"\n";

    auto st = std::chrono::system_clock::now();
    auto outputs = embed->onForward({input_var});
    auto et = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(et - st);
    printf("# 1. embedding times: %f ms\n", duration.count() * 1e-3);

    auto depth = _Convert(outputs[0], NCHW);

    int h= depth->getInfo()->dim[2];
    int w= depth->getInfo()->dim[3];
    // resize to (length, length)
    cv::Mat mask_cv(h, w, CV_32FC1);
    const float* dataPtr= depth->readMap<float>();
    memcpy(mask_cv.data, dataPtr, h*w* sizeof(float));
    cv::Rect roi(0, 0, new_w, new_h);
    cv::Mat mask_crop = mask_cv(roi);
    // resize to (origin_w, origin_h)
    cv::resize(mask_crop, mask_crop, cv::Size(origin_w, origin_h));
    double minValue, maxValue;
    cv::minMaxLoc(mask_crop, &minValue, &maxValue);
    cv::subtract(mask_crop, minValue, mask_crop);
    mask_crop= mask_crop/ (maxValue-minValue) *255.0 ;
    mask_crop.convertTo(mask_crop, CV_8UC1);
    cv::Mat depth_color;
    cv::applyColorMap(mask_crop, depth_color, cv::COLORMAP_INFERNO);
    
    cv::imwrite("res_cpp.jpg", depth_color);

    return 0;
}
