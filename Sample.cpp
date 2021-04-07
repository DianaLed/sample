// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <string>
#include <iterator>
#include <samples/common.hpp>

#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>
#include <samples/classification_results.h>

using namespace InferenceEngine;

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
#define tcout std::wcout
#define file_name_t std::wstring
#define imread_t imreadW
#define ClassificationResult_t ClassificationResultW
#else
#define tcout std::cout
#define file_name_t std::string
#define imread_t cv::imread
#define ClassificationResult_t ClassificationResult
#endif

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
cv::Mat imreadW(std::wstring input_image_path) {
    cv::Mat image;
    std::ifstream input_image_stream;
    input_image_stream.open(
        input_image_path.c_str(),
        std::iostream::binary | std::ios_base::ate | std::ios_base::in);
    if (input_image_stream.is_open()) {
        if (input_image_stream.good()) {
            input_image_stream.seekg(0, std::ios::end);
            std::size_t file_size = input_image_stream.tellg();
            input_image_stream.seekg(0, std::ios::beg);
            std::vector<char> buffer(0);
            std::copy(
                std::istreambuf_iterator<char>(input_image_stream),
                std::istreambuf_iterator<char>(),
                std::back_inserter(buffer));
            image = cv::imdecode(cv::Mat(1, file_size, CV_8UC1, &buffer[0]), cv::IMREAD_COLOR);
        } else {
            tcout << "Input file '" << input_image_path << "' processing error" << std::endl;
        }
        input_image_stream.close();
    } else {
        tcout << "Unable to read input file '" << input_image_path << "'" << std::endl;
    }
    return image;
}

std::string simpleConvert(const std::wstring & wstr) {
    std::string str;
    for (auto && wc : wstr)
        str += static_cast<char>(wc);
    return str;
}

int wmain(int argc, wchar_t *argv[]) {
#else

int main(int argc, char *argv[]) {
#endif
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (argc != 4) {
            tcout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device_name>" << std::endl; //прописать сеть, характеристики
            return EXIT_FAILURE;
        }

        const file_name_t input_model{argv[1]};
        const file_name_t input_image_path{argv[2]};
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const std::string device_name = simpleConvert(argv[3]);
#else
        const std::string device_name{argv[3]};
#endif
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine instance -------------------------------------
        Core ie;//обьявление ядра
        // -----------------------------------------------------------------------------------------------------

        // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        CNNNetwork network = ie.ReadNetwork(input_model); //cnn network-сверточные нейр сети //cxbnsdftn vjltkm bp ajhvfnf ir
        if (network.getOutputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 output only"); //с выходом (одно изображение)
        if (network.getInputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");   // один вход (итог)
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        InputInfo::Ptr input_info = network.getInputsInfo().begin()->second; //получить всю инфу nchw
        std::string input_name = network.getInputsInfo().begin()->first; //получить имя

        /* Mark input as resizable by setting of a resize algorithm.
         * In this case we will be able to set an input blob of any shape to an infer request.
         * Resize and layout conversions are executed automatically during inference */
        input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR); // картинку перересайзнули
        input_info->setLayout(Layout::NHWC); //подготовили под nhwc
        input_info->setPrecision(Precision::U8); //тип данных изображний 

        // --------------------------- Prepare output blobs ----------------------------------------------------
        DataPtr output_info = network.getOutputsInfo().begin()->second; //на выходе численные данные
        std::string output_name = network.getOutputsInfo().begin()->first; //получение имя выхода

        output_info->setPrecision(Precision::FP32); //описали тип данных пресижена //все описали
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        ExecutableNetwork executable_network = ie.LoadNetwork(network, device_name); //cpu //загрузка сети в сам плагин, проверка предыдущих строк
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest(); //создаются потоки
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        /* Read input image to a blob and set it to an infer request without resize and layout conversions. */
        cv::Mat image = imread_t(input_image_path);  //считываем картинку, и переводим в blob
        Blob::Ptr imgBlob = wrapMat2Blob(image);  // just wrap Mat data by Blob::Ptr without allocating of new memory
        infer_request.SetBlob(input_name, imgBlob);  // infer_request accepts input blob of any size
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference --------------------------------------------------------
        /* Running the request synchronously */
        infer_request.Infer(); //запускаем сеть
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output ------------------------------------------------------
        Blob::Ptr output = infer_request.GetBlob(output_name);
        // Print classification results
        ClassificationResult_t classificationResult(output, {input_image_path}); //обработка данных 
        classificationResult.print();
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool" << std::endl;
    return EXIT_SUCCESS;
}