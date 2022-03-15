// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>

#include "local_filesystem.h"

#define my_strtol strtol
#define my_strrchr strrchr
#define my_strcasecmp strcasecmp
#define my_strdup strdup

#include <inference_engine.hpp>
#include <onnxruntime_cxx_api.h>
#include <condition_variable>
#include <fstream>
#include <ctime>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include "dlib/opencv/cv_image.h"
#include "dlib/opencv/to_open_cv.h"
#include "FaceAligner.h"

using namespace std::chrono;
using namespace FaceAlignment;

template <typename T>
T vectorProduct(const std::vector<T>& v) {
	return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
	os << "[";
	for (int i = 0; i < v.size(); ++i) {
		os << v[i];
		if (i != v.size() - 1) {
			os << ", ";
		}
	}
	os << "]";
	return os;
}



class GenericONNXModel {
private:
	static void VerifyInputOutputCount(Ort::Session& session, size_t input_count, size_t output_count) {
		size_t count = session.GetInputCount();
		assert(count == input_count);
		count = session.GetOutputCount();
		assert(count == output_count);
	}

	Ort::Session session_{ nullptr };

	// std::mutex m_;
	std::vector<char*> input_names_;
	std::vector<char*> output_names_;

	Ort::Env& env_;
	const TCharString model_path_;
	system_clock::time_point start_time_;
	std::vector<std::vector<int64_t>> input_dims_;
	std::vector<std::vector<int64_t>> output_dims_;

public:
	std::vector<char*> GetInputNames() const { return input_names_; }
	std::vector<char*> GetOutputNames() const { return output_names_; }
	int GetInputCount() const { return input_names_.size(); }
	int GetOutputCount() const { return output_names_.size(); }

	std::vector<int64_t> GetInputDims(size_t n) const {
		if (n < input_dims_.size())
			return input_dims_[n];
		else
			throw std::out_of_range("input_dims_");
	}

	std::vector<int64_t> GetOutputDims(size_t n) const {
		if (n < output_dims_.size())
			return output_dims_[n];
		else
			throw std::out_of_range("output_dims_");
	}

	~GenericONNXModel() {
		for (std::vector<char*>::const_iterator it = input_names_.begin(); it != input_names_.end(); ++it) free(*it);
		for (std::vector<char*>::const_iterator it = output_names_.begin(); it != output_names_.end(); ++it) free(*it);
	}

	void CreateSession(bool use_cuda) {
		Ort::SessionOptions session_options;

		if (use_cuda) {
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
		}
		session_ = Ort::Session(env_, model_path_.c_str(), session_options);
	}

	GenericONNXModel(Ort::Env& env, const TCharString& model_path, int n_input, int n_output, bool use_cuda) : env_(env), model_path_(model_path) {
		std::cout << "Loading model...";
		system_clock::time_point loading_start_time = system_clock::now();
		CreateSession(use_cuda);
		std::cout << "loaded [";
		auto elapsed = system_clock::now() - loading_start_time;
		auto secs = duration_cast<seconds>(elapsed).count();
		std::cout << secs << " seconds]." << std::endl;
		VerifyInputOutputCount(session_, n_input, n_output);
		Ort::AllocatorWithDefaultOptions ort_alloc;
		{
			int n_input = session_.GetInputCount();
			for (int i = 0; i < n_input; ++i) {
				char* t = session_.GetInputName(i, ort_alloc);
				char* input_name = my_strdup(t);
				input_names_.push_back(input_name);
				ort_alloc.Free(t);
			}

			int n_output = session_.GetOutputCount();
			for (int i = 0; i < n_output; ++i) {
				char* t = session_.GetOutputName(i, ort_alloc);
				char* output_name_ = my_strdup(t);
				output_names_.push_back(output_name_);
				ort_alloc.Free(t);
			}
		}

		for (int i = 0; i < input_names_.size(); ++i) {
			Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(i);
			auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
			std::vector<int64_t> input_dim = inputTensorInfo.GetShape();
			ONNXTensorElementDataType tensorType = inputTensorInfo.GetElementType();
			std::cout << "Input dimensions for " << i << "-th input: " << input_dim << std::endl;
			if (input_dim[0] < 0) input_dim[0] = 1;
			input_dims_.push_back(input_dim);
		}

		for (int i = 0; i < output_names_.size(); ++i) {
			Ort::TypeInfo outputTypeInfo = session_.GetOutputTypeInfo(i);
			auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
			std::vector<int64_t> output_dim_ = outputTensorInfo.GetShape();
			ONNXTensorElementDataType tensorType = outputTensorInfo.GetElementType();
			std::cout << "Output dimensions for " << i << "-th output: " << output_dim_ << std::endl;
			if (output_dim_[0] < 0) output_dim_[0] = 1;
			output_dims_.push_back(output_dim_);
		}

		start_time_ = system_clock::now();
	}

	void alloc_input_buffers(std::vector<std::vector<float>>& buf) {
		buf.clear();
		for (int i = 0; i < input_names_.size(); ++i) {
			size_t inputTensorSize = vectorProduct(input_dims_[i]);
			std::vector<float> temp(inputTensorSize, 0);
			buf.push_back(temp);
		}
	}

	void create_input_tensors(std::vector<std::vector<float>>& buf, std::vector<Ort::Value>& input_tensors) {
		Ort::MemoryInfo memInfo =
			Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		for (int i = 0; i < input_names_.size(); ++i) {
			input_tensors.push_back(Ort::Value::CreateTensor<float>(memInfo, buf[i].data(), buf[i].size(),
				input_dims_[i].data(), input_dims_[i].size()));
		}
	}

	void alloc_output_buffers(std::vector<std::vector<float>>& buf) {
		buf.clear();
		for (int i = 0; i < output_names_.size(); ++i) {
			size_t outputTensorSize = vectorProduct(output_dims_[i]);
			std::vector<float> temp(outputTensorSize, 0);
			buf.push_back(temp);
		}
	}

	void create_output_tensors(std::vector<std::vector<float>>& buf, std::vector<Ort::Value>& output_tensors) {
		Ort::MemoryInfo memInfo =
			Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		for (int i = 0; i < output_names_.size(); ++i) {
			output_tensors.push_back(Ort::Value::CreateTensor<float>(memInfo, buf[i].data(), buf[i].size(),
				output_dims_[i].data(), output_dims_[i].size()));
		}
	}

	void run(std::vector<Ort::Value>& input_tensors, std::vector<Ort::Value>& output_tensors) {
		session_.Run(Ort::RunOptions{ nullptr }, input_names_.data(), input_tensors.data(), input_tensors.size(),
			output_names_.data(), output_tensors.data(), output_tensors.size());
	}

};  // End of class Decoder

int run_e4e(int argc, ORTCHAR_T* argv[]) {
	if (argc < 6) return -1;
	TCharString data_dir = argv[1];
	TCharString e4e_path = argv[2];
	TCharString decoder_path = argv[3];
	TCharString decoder_path_2 = argv[4];
	std::vector<TCharString> image_file_paths;

	LoopDir(data_dir, [&data_dir, &image_file_paths](const ORTCHAR_T* filename, OrtFileType filetype) -> bool {
		if (filetype != OrtFileType::TYPE_REG) return true;
		if (filename[0] == '.') return true;
		const ORTCHAR_T* p = my_strrchr(filename, '.');
		if (p == nullptr) return true;
		// as we tested filename[0] is not '.', p should larger than filename
		assert(p > filename);
		if (my_strcasecmp(p, ORT_TSTR(".png")) != 0 && my_strcasecmp(p, ORT_TSTR(".JPG")) != 0) return true;
		TCharString v(data_dir);
		v.append(1, '/');
		v.append(filename);
		image_file_paths.emplace_back(v);
		return true;
		});

	/////////////////// Prepare the face aligner
	std::string dlib_shape_predictor_file =
		"/media/kent/DISK2/VMProjects/E2ID_plus/shape_predictor_68_face_landmarks.dat";
	shape_predictor sp;
	try {
		dlib::deserialize(dlib_shape_predictor_file) >> sp;
	}
	catch (std::exception& ex) {
		std::cout << "Error: " << ex.what() << std::endl;
	}
	frontal_face_detector face_detector;
	face_detector = get_frontal_face_detector();

	//////////////////// For demo, only load the first image
	std::string strTo(image_file_paths[0].size(), 0);
	array2d<rgb_pixel> img;
	load_image(img, image_file_paths[0]);
	std::vector<dlib::rectangle> faces = face_detector(img);
	if (faces.size() == 0) {
		std::cout << "No faces detected. Exited." << std::endl;
		std::exit(-1);
	}
	FaceAligner face_aligner(sp, 256);
	array2d<rgb_pixel> aligned = face_aligner.align(img, faces[0], AlignmentType::CELEBA);
	dlib::save_png(aligned, "aligned.png");

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");
	GenericONNXModel e4e(env, e4e_path, 1, 1, false); // Load the encoder
	cv::Mat imageRGB, resizedRGB, resizedBGR, inputImage;  // @Bai 定义了 input 图片
	imageRGB = dlib::toMat(aligned);

	//////////////////////// Image normalization
	cv::resize(imageRGB, resizedRGB, cv::Size(e4e.GetInputDims(0)[2], e4e.GetInputDims(0)[3]),
		cv::InterpolationFlags::INTER_LINEAR);
	resizedRGB.convertTo(inputImage, CV_32F, 1.0 / 255);
	cv::Mat channels[3];
	cv::split(inputImage, channels);
	channels[0] = (channels[0] - 0.5) * 2;
	channels[1] = (channels[1] - 0.5) * 2;
	channels[2] = (channels[2] - 0.5) * 2;
	cv::merge(channels, 3, inputImage);
	cv::Mat imageCHW;
	cv::dnn::blobFromImage(inputImage, imageCHW);   // HWC to CHW

	//////////////////////// Prepare data for forward inference of the encoder
	std::vector<std::vector<float>> input_buf_e4e;
	std::vector<std::vector<float>> output_buf_e4e;
	std::vector<Ort::Value> input_tensors, output_tensors;
	//e4e.alloc_input_buffers(input_buf_e4e);
	std::vector<float> tmp_buf;
	tmp_buf.assign(imageCHW.begin<float>(), imageCHW.end<float>());
	input_buf_e4e.push_back(tmp_buf);
	e4e.alloc_output_buffers(output_buf_e4e);
	e4e.create_input_tensors(input_buf_e4e, input_tensors);
	e4e.create_output_tensors(output_buf_e4e, output_tensors);

	e4e.run(input_tensors, output_tensors); // Image -> Latent

	clock_t startTime, endTime;

	/**
	 * @Bai OpenVINO Conversion [e4e]
	 */

	// Step 1. Initialize inference engine core
	InferenceEngine::Core core;
	InferenceEngine::CNNNetwork network_e4e;
	InferenceEngine::ExecutableNetwork executable_network_e4e;

	startTime = clock();
	// Step 2. Read a model
	network_e4e = core.ReadNetwork("/media/kent/DISK2/VMProjects/super_realism/e4e_onnx/e4e_vino/models/e4e.xml");

	// Step 3. Configure input & output

	// Prepare input blobs
	auto info = network_e4e.getInputsInfo();
	InferenceEngine::InputInfo::Ptr input_info = network_e4e.getInputsInfo().begin()->second;
	std::string input_name = network_e4e.getInputsInfo().begin()->first;

	// Prepare output blobs
	std::string output_name = network_e4e.getOutputsInfo().begin()->first;

	// Step 4. Loading a model to the device
	executable_network_e4e = core.LoadNetwork(network_e4e, "CPU");
	endTime = clock();

	// Step 5. Create an infer request
	auto infer_request = executable_network_e4e.CreateInferRequest();

	// Step 6. Prepare input
	InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, {1, 3, 256, 256}, InferenceEngine::Layout::NCHW);
	InferenceEngine::Blob::Ptr imgBlob = InferenceEngine::make_shared_blob<float>(tDesc, (float *)imageCHW.data);
	infer_request.SetBlob(input_name, imgBlob);

	// Step 7. Do inference
	infer_request.Infer();

	// Step 8. Process output
	InferenceEngine::Blob::Ptr output_e4e = infer_request.GetBlob(output_name);

	// 获取 e4e output data 方法1:
	InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(infer_request.GetBlob(output_name));
	auto lmoHolder = moutput->rmap();
	const auto output_data = lmoHolder.as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
	std::vector<float> out_e4e(1*1*18*512);
	for (int i = 0; i < 1*1*18*512; i++) {
		out_e4e[i] = static_cast<float>(output_data[i]);
	}


	std::cout << "[vino] e4e done. Time: " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

	////////////////////// First part of StyleGAN2 
	GenericONNXModel decoder_512(env, decoder_path, 1, 3, false);
	// std::vector<std::vector<float>> decoder_input_buf;
	std::vector<std::vector<float>> decoder_output_buf;
	std::vector<Ort::Value> decoder_input_tensors, decoder_output_tensors;

	decoder_512.create_input_tensors(output_buf_e4e, decoder_input_tensors); // Use the output of the encoder as the input of the decoder
	decoder_512.alloc_output_buffers(decoder_output_buf);
	decoder_512.create_output_tensors(decoder_output_buf, decoder_output_tensors);

	decoder_512.run(decoder_input_tensors, decoder_output_tensors); // Latent -> Image

	/**
	 * @Bai OpenVINO Conversion [d512]
	 */


	InferenceEngine::CNNNetwork network_d512;
	InferenceEngine::ExecutableNetwork executable_network_d512;

	startTime = clock();
	// Step 2. Read a model
	network_d512 = core.ReadNetwork("/media/kent/DISK2/VMProjects/super_realism/e4e_onnx/e4e_vino/models/decoder4e4e_1.xml");

	// Step 3. Configure input & output

	// Step 4. Loading a model to the device
	executable_network_d512 = core.LoadNetwork(network_d512, "CPU");
	endTime = clock();

	// Step 5. Create an infer request
	auto infer_request_d512 = executable_network_d512.CreateInferRequest();

	// Step 6. Prepare input
	infer_request_d512.SetBlob("input", output_e4e);

	// Step 7. Do inference
	infer_request_d512.Infer();

	// Step 8. Process output
	InferenceEngine::Blob::Ptr output_d512_latent = infer_request_d512.GetBlob("latent");
	InferenceEngine::Blob::Ptr output_d512_features = infer_request_d512.GetBlob("features");
	InferenceEngine::Blob::Ptr output_d512_image = infer_request_d512.GetBlob("image");

	std::cout << "[vino] d512 done. Time: " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

	//////////////////////// Second part of StyleGAN2
	GenericONNXModel decoder_1024(env, decoder_path_2, 3, 1, false);
	std::vector<std::vector<float>> decoder1024_output_buf;
	std::vector<Ort::Value> decoder1024_input_tensors, decoder1024_output_tensors;
	decoder_1024.create_input_tensors(decoder_output_buf, decoder1024_input_tensors);
	decoder_1024.alloc_output_buffers(decoder1024_output_buf);
	decoder_1024.create_output_tensors(decoder1024_output_buf, decoder1024_output_tensors);

	decoder_1024.run(decoder1024_input_tensors, decoder1024_output_tensors); // 512 x 512 -> 1024 x 1024

	/**
	 * @Bai OpenVINO Conversion [d1024]
	 */


	InferenceEngine::CNNNetwork network_d1024;
	InferenceEngine::ExecutableNetwork executable_network_d1024;

	startTime = clock();
	// Step 2. Read a model
	network_d1024 = core.ReadNetwork("/media/kent/DISK2/VMProjects/super_realism/e4e_onnx/e4e_vino/models/decoder4e4e_2.xml");

	// Step 3. Configure input & output

	// Step 4. Loading a model to the device
	executable_network_d1024 = core.LoadNetwork(network_d1024, "CPU");
	endTime = clock();

	// Step 5. Create an infer request
	auto infer_request_d1024 = executable_network_d1024.CreateInferRequest();

	// Step 6. Prepare input
	infer_request_d1024.SetBlob("input_1", output_d512_latent);
	infer_request_d1024.SetBlob("input_2", output_d512_features);
	infer_request_d1024.SetBlob("input_3", output_d512_image);

	// Step 7. Do inference
	infer_request_d1024.Infer();

	// Step 8. Process output
	InferenceEngine::Blob::Ptr output_d1024 = infer_request_d1024.GetBlob("image");
	float* buff = static_cast<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>(output_d1024->buffer());

	std::cout << "[vino] d1024 done. Time: " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

	///////////////////// Convert the output tensor to an image
	const std::vector<int64_t>& output_shape = decoder_1024.GetOutputDims(0);
	std::vector<int> output_shape_(output_shape.begin(), output_shape.end());
	// cv::Mat output_blob(output_shape_.size(), output_shape_.data(), CV_32FC1, decoder1024_output_buf[0].data()); // Onnx output
	cv::Mat output_blob(output_shape_.size(), output_shape_.data(), CV_32FC1, output_d1024->buffer()); // Openvino output
	std::vector<cv::Mat> images;
	cv::dnn::imagesFromBlob(output_blob, images);
	std::cout << "nb of images: " << images.size() << std::endl;
	std::cout << "channels: " << images[0].channels() << std::endl;
	std::cout << "size: " << images[0].size() << std::endl;

	cv::split(images[0], channels);
	for (int i = 0; i < 3; ++i) {
		channels[i] = (channels[i] + 1) / 2;
		channels[i].setTo(0, channels[i] < 0);
		channels[i].setTo(1, channels[i] > 1);
		channels[i] = channels[i] * 255;
		channels[i].convertTo(channels[i], CV_8UC1);
	}
	cv::Mat image;
	cv::merge(channels, 3, image);
	cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
	cv::imwrite("onnx.png", image);

	return 0;
}

int main(int argc, ORTCHAR_T * argv[]) {
	int ret = -1;
	try {
		// ret = real_main(argc, argv);
		ret = run_e4e(argc, argv);
	}
	catch (const std::exception& ex) {
		fprintf(stderr, "%s\n", ex.what());
	}
	return ret;
}
