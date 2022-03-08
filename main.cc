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

#include "local_filesystem.h"

#ifdef _WIN32
#define my_strtol wcstol
#define my_strrchr wcsrchr
#define my_strcasecmp _wcsicmp
#define my_strdup _strdup
#else
#define my_strtol strtol
#define my_strrchr strrchr
#define my_strcasecmp strcasecmp
#define my_strdup strdup
#endif

#include <onnxruntime_cxx_api.h>
#include <condition_variable>
#include <fstream>
#ifdef _WIN32
#include <atlbase.h>
#endif

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

class Hypernet {
private:
	static std::vector<std::string> ReadFileToVec(const TCharString& file_path, size_t expected_line_count) {
		std::ifstream ifs(file_path);
		if (!ifs) {
			throw std::runtime_error("open file failed");
		}
		std::string line;
		std::vector<std::string> labels;
		while (std::getline(ifs, line)) {
			if (!line.empty()) labels.push_back(line);
		}
		if (labels.size() != expected_line_count) {
			std::ostringstream oss;
			oss << "line count mismatch, expect " << expected_line_count << " from " << file_path.c_str() << ", got "
				<< labels.size();
			throw std::runtime_error(oss.str());
		}
		return labels;
	}

	// input file name has pattern like:
	//"C:\tools\imagnet_validation_data\ILSVRC2012_val_00000001.JPEG"
	//"C:\tools\imagnet_validation_data\ILSVRC2012_val_00000002.JPEG"
	static int ExtractImageNumberFromFileName(const TCharString& image_file) {
		size_t s = image_file.rfind('.');
		if (s == std::string::npos) throw std::runtime_error("illegal filename");
		size_t s2 = image_file.rfind('_');
		if (s2 == std::string::npos) throw std::runtime_error("illegal filename");

		const ORTCHAR_T* start_ptr = image_file.c_str() + s2 + 1;
		const ORTCHAR_T* endptr = nullptr;
		long value = my_strtol(start_ptr, (ORTCHAR_T**)&endptr, 10);
		if (start_ptr == endptr || value > INT32_MAX || value <= 0) throw std::runtime_error("illegal filename");
		return static_cast<int>(value);
	}

	static void VerifyInputOutputCount(Ort::Session& session) {
		size_t count = session.GetInputCount();
		assert(count == 2);
		count = session.GetOutputCount();
		assert(count == 15);
	}

	Ort::Session session_{ nullptr };
	const int output_class_count_ = 1001;
	std::vector<std::string> labels_;
	std::vector<std::string> validation_data_;
	std::atomic<int> top_1_correct_count_;
	std::atomic<int> finished_count_;
	int image_size_;

	std::mutex m_;

	// char* output_name_ = nullptr;
	std::vector<char*> input_names_;
	std::vector<char*> output_names_;

	Ort::Env& env_;
	const TCharString model_path_;
	system_clock::time_point start_time_;
	std::vector<std::vector<int64_t>> input_dims_;
	std::vector<std::vector<int64_t>> output_dims_;

public:
	int GetImageSize() const { return image_size_; }
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

	~Hypernet() {
		for (std::vector<char*>::const_iterator it = input_names_.begin(); it != input_names_.end(); ++it) free(*it);
		for (std::vector<char*>::const_iterator it = output_names_.begin(); it != output_names_.end(); ++it) free(*it);
	}

	void PrintResult() {
		if (finished_count_ == 0) return;
		printf("Top-1 Accuracy %f\n", ((float)top_1_correct_count_.load() / finished_count_));
	}

	void CreateSession() {
		Ort::SessionOptions session_options;
#ifdef USE_CUDA
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif
		session_ = Ort::Session(env_, model_path_.c_str(), session_options);
	}

	Hypernet(Ort::Env& env, const TCharString& model_path, size_t input_image_count)
		: env_(env), model_path_(model_path) {
		CreateSession();
		std::cout << "Hypernet ONNX model loaded." << std::endl;
		VerifyInputOutputCount(session_);
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

	void alloc_input_buffers(std::vector<float>& buf1, std::vector<float>& buf2) {
		size_t inputTensorSize = vectorProduct(input_dims_[0]);
		buf1.resize(inputTensorSize, 0);
		inputTensorSize = vectorProduct(input_dims_[1]);
		buf2.resize(inputTensorSize, 0);
	}

	void create_input_tensors(std::vector<float>& buf1, std::vector<float>& buf2,
		std::vector<Ort::Value>& input_tensors) {
		Ort::MemoryInfo memInfo =
			Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		input_tensors.push_back(Ort::Value::CreateTensor<float>(memInfo, buf1.data(), buf1.size(), input_dims_[0].data(),
			input_dims_[0].size()));
		input_tensors.push_back(Ort::Value::CreateTensor<float>(memInfo, buf2.data(), buf2.size(), input_dims_[1].data(),
			input_dims_[1].size()));
	}

	void alloc_output_buffers(std::vector<int64_t>& buf1, std::vector<std::vector<float>>& buf2) {
		size_t outputTensorSize = vectorProduct(output_dims_[0]);
		buf1.resize(outputTensorSize, 0);
		buf2.clear();
		for (int i = 1; i < output_names_.size(); ++i) {
			size_t outputTensorSize = vectorProduct(output_dims_[i]);
			std::vector<float> temp(outputTensorSize, 0);
			buf2.push_back(temp);
		}
	}

	void create_output_tensors(std::vector<int64_t>& buf1, std::vector<std::vector<float>>& buf2,
		std::vector<Ort::Value>& output_tensors) {
		Ort::MemoryInfo memInfo =
			Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		output_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memInfo, buf1.data(), buf1.size(),
			output_dims_[0].data(), output_dims_[0].size()));
		for (int i = 1; i < output_names_.size(); ++i) {
			output_tensors.push_back(Ort::Value::CreateTensor<float>(memInfo, buf2[i - 1].data(), buf2[i - 1].size(),
				output_dims_[i].data(), output_dims_[i].size()));
		}
	}

	void run(std::vector<Ort::Value>& input_tensors, std::vector<Ort::Value>& output_tensors) {
		session_.Run(Ort::RunOptions{ nullptr }, input_names_.data(), input_tensors.data(), input_tensors.size(),
			output_names_.data(), output_tensors.data(), output_tensors.size());
	}
};

class Decoder {
private:
	// input file name has pattern like:
	//"C:\tools\imagnet_validation_data\ILSVRC2012_val_00000001.JPEG"
	//"C:\tools\imagnet_validation_data\ILSVRC2012_val_00000002.JPEG"
	static int ExtractImageNumberFromFileName(const TCharString& image_file) {
		size_t s = image_file.rfind('.');
		if (s == std::string::npos) throw std::runtime_error("illegal filename");
		size_t s2 = image_file.rfind('_');
		if (s2 == std::string::npos) throw std::runtime_error("illegal filename");

		const ORTCHAR_T* start_ptr = image_file.c_str() + s2 + 1;
		const ORTCHAR_T* endptr = nullptr;
		long value = my_strtol(start_ptr, (ORTCHAR_T**)&endptr, 10);
		if (start_ptr == endptr || value > INT32_MAX || value <= 0) throw std::runtime_error("illegal filename");
		return static_cast<int>(value);
	}

	static void VerifyInputOutputCount(Ort::Session& session) {
		size_t count = session.GetInputCount();
		assert(count == 15);
		count = session.GetOutputCount();
		assert(count == 2);
	}

	Ort::Session session_{ nullptr };
	int image_size_;
	// std::mutex m_;
	std::vector<char*> input_names_;
	std::vector<char*> output_names_;

	Ort::Env& env_;
	const TCharString model_path_;
	system_clock::time_point start_time_;
	std::vector<std::vector<int64_t>> input_dims_;
	std::vector<std::vector<int64_t>> output_dims_;

public:
	int GetImageSize() const { return image_size_; }
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

	~Decoder() {
		for (std::vector<char*>::const_iterator it = input_names_.begin(); it != input_names_.end(); ++it) free(*it);
		for (std::vector<char*>::const_iterator it = output_names_.begin(); it != output_names_.end(); ++it) free(*it);
	}

	void CreateSession() {
		Ort::SessionOptions session_options;
#ifdef USE_CUDA
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif
		session_ = Ort::Session(env_, model_path_.c_str(), session_options);
	}

	Decoder(Ort::Env& env, const TCharString& model_path, size_t input_image_count) : env_(env), model_path_(model_path) {
		std::cout << "Loading StyleGAN2 model...";
		CreateSession();
		std::cout << "loaded." << std::endl;
		VerifyInputOutputCount(session_);
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

	// TODO: remove the slash at the end of data_dir string
	LoopDir(data_dir, [&data_dir, &image_file_paths](const ORTCHAR_T* filename, OrtFileType filetype) -> bool {
		if (filetype != OrtFileType::TYPE_REG) return true;
		if (filename[0] == '.') return true;
		const ORTCHAR_T* p = my_strrchr(filename, '.');
		if (p == nullptr) return true;
		// as we tested filename[0] is not '.', p should larger than filename
		assert(p > filename);
		if (my_strcasecmp(p, ORT_TSTR(".png")) != 0 && my_strcasecmp(p, ORT_TSTR(".JPG")) != 0) return true;
		TCharString v(data_dir);
#ifdef _WIN32
		v.append(1, '\\');
#else
		v.append(1, '/');
#endif
		v.append(filename);
		image_file_paths.emplace_back(v);
		return true;
		});

	/////////////////// Prepare the face aligner
	std::string dlib_shape_predictor_file =
		"D:/VisionMetric/efit6/Photo2Fit/hourglass_model/shape_predictor_68_face_landmarks.dat";
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
	WideCharToMultiByte(CP_UTF8, 0, image_file_paths[0].data(), (int)image_file_paths[0].size(), &strTo[0],
		image_file_paths[0].size(), NULL, NULL);
	array2d<rgb_pixel> img;
	load_image(img, strTo);
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
	cv::Mat imageRGB, resizedRGB, resizedBGR, inputImage;
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

	////////////////////// First part of StyleGAN2 
	GenericONNXModel decoder_512(env, decoder_path, 1, 3, false);
	// std::vector<std::vector<float>> decoder_input_buf;
	std::vector<std::vector<float>> decoder_output_buf;
	std::vector<Ort::Value> decoder_input_tensors, decoder_output_tensors;

	decoder_512.create_input_tensors(output_buf_e4e, decoder_input_tensors); // Use the output of the encoder as the input of the decoder
	decoder_512.alloc_output_buffers(decoder_output_buf);
	decoder_512.create_output_tensors(decoder_output_buf, decoder_output_tensors);

	decoder_512.run(decoder_input_tensors, decoder_output_tensors); // Latent -> Image

	//////////////////////// Second part of StyleGAN2
	GenericONNXModel decoder_1024(env, decoder_path_2, 3, 1, false);
	std::vector<std::vector<float>> decoder1024_output_buf;
	std::vector<Ort::Value> decoder1024_input_tensors, decoder1024_output_tensors;
	decoder_1024.create_input_tensors(decoder_output_buf, decoder1024_input_tensors);
	decoder_1024.alloc_output_buffers(decoder1024_output_buf);
	decoder_1024.create_output_tensors(decoder1024_output_buf, decoder1024_output_tensors);

	decoder_1024.run(decoder1024_input_tensors, decoder1024_output_tensors); // 512 x 512 -> 1024 x 1024

	///////////////////// Convert the output tensor to an image
	std::vector<int64_t>& output_shape = decoder_1024.GetOutputDims(0);
	std::vector<int> output_shape_(output_shape.begin(), output_shape.end());
	cv::Mat output_blob(output_shape_.size(), output_shape_.data(), CV_32FC1, decoder1024_output_buf[0].data());
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

//int real_main(int argc, ORTCHAR_T* argv[]) {
//  if (argc < 5) return -1;
//  std::vector<TCharString> image_file_paths;
//  TCharString data_dir = argv[1];
//  TCharString model_path = argv[2];
//  TCharString decoder_path = argv[3];
//  // imagenet_lsvrc_2015_synsets.txt
//  const int batch_size = std::stoi(argv[4]);
//
//  // TODO: remove the slash at the end of data_dir string
//  LoopDir(data_dir, [&data_dir, &image_file_paths](const ORTCHAR_T* filename, OrtFileType filetype) -> bool {
//    if (filetype != OrtFileType::TYPE_REG) return true;
//    if (filename[0] == '.') return true;
//    const ORTCHAR_T* p = my_strrchr(filename, '.');
//    if (p == nullptr) return true;
//    // as we tested filename[0] is not '.', p should larger than filename
//    assert(p > filename);
//    if (my_strcasecmp(p, ORT_TSTR(".png")) != 0 && my_strcasecmp(p, ORT_TSTR(".JPG")) != 0) return true;
//    TCharString v(data_dir);
//#ifdef _WIN32
//    v.append(1, '\\');
//#else
//		v.append(1, '/');
//#endif
//    v.append(filename);
//    image_file_paths.emplace_back(v);
//    return true;
//  });
//
//  std::vector<uint8_t> data;
//  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");
//
//  Hypernet v(env, model_path, image_file_paths.size());
//
//  // Prepare data for forward inference
//  std::vector<float> input_buf1, input_buf2;
//  std::vector<int64_t> output_buf1;
//  std::vector<std::vector<float>> output_buf2;
//  std::vector<Ort::Value> input_tensors, output_tensors;
//
//  v.alloc_input_buffers(input_buf1, input_buf2);
//  v.alloc_output_buffers(output_buf1, output_buf2);
//  v.create_input_tensors(input_buf1, input_buf2, input_tensors);
//  v.create_output_tensors(output_buf1, output_buf2, output_tensors);
//
//  v.run(input_tensors, output_tensors);
//
//  std::ofstream ofs;
//  ofs.open("weight_deltas.txt");
//  for (std::vector<float>::const_iterator it = output_buf2[0].begin(); it != output_buf2[0].end(); ++it) {
//    ofs << std::setprecision(6) << *it << ",";
//  }
//  ofs.close();
//
//  // StyleGAN2
//  Decoder decoder(env, decoder_path, image_file_paths.size());
//  std::vector<std::vector<float>> decoder_input_buf;
//  std::vector<std::vector<float>> decoder_output_buf;
//  std::vector<Ort::Value> decoder_input_tensors, decoder_output_tensors;
//
//  decoder.alloc_input_buffers(decoder_input_buf);
//  assert(decoder_input_buf.size() == output_buf2.size() + 1);
//  for (int i = 0; i < output_buf2.size(); ++i) {
//    assert(decoder_input_buf[i + 1].size() == output_buf2[i].size());
//    decoder_input_buf[i + 1] = output_buf2[i];
//  }
//  output_buf2.clear();
//  decoder.create_input_tensors(decoder_input_buf, decoder_input_tensors);
//  decoder.alloc_output_buffers(decoder_output_buf);
//  decoder.create_output_tensors(decoder_output_buf, decoder_output_tensors);
//
//  decoder.run(decoder_input_tensors, decoder_output_tensors);
//
//  // Convert the output tensor to an image
//  std::vector<int64_t>& output_shape = decoder.GetOutputDims(0);
//  // std::vector<int> output_shape_(output_shape.size(), 0);
//  // for (int i = 0; i < output_shape.size(); ++i) output_shape_[i] = output_shape[i];
//  std::vector<int> output_shape_(output_shape.begin(), output_shape.end());
//  cv::Mat output_blob(output_shape_.size(), output_shape_.data(), CV_32FC1, decoder_output_buf[0].data());
//  std::vector<cv::Mat> images;
//  cv::dnn::imagesFromBlob(output_blob, images);
//  std::cout << "nb of images: " << images.size() << std::endl;
//  std::cout << "channels: " << images[0].channels() << std::endl;
//  std::cout << "size: " << images[0].size() << std::endl;
//  cv::Mat channels[3];
//  cv::split(images[0], channels);
//  for (int i = 0; i < 3; ++i) {
//    channels[i] = (channels[i] + 1) / 2;
//    channels[i].setTo(0, channels[i] < 0);
//    channels[i].setTo(1, channels[i] > 1);
//    channels[i] = channels[i] * 255;
//    channels[i].convertTo(channels[i], CV_8UC1);
//  }
//  cv::Mat image;
//  cv::merge(channels, 3, image);
//  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
//  cv::imwrite("onnx.png", image);
//
//  return 0;
//}
#ifdef _WIN32
int wmain(int argc, ORTCHAR_T* argv[]) {
	HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
	if (!SUCCEEDED(hr)) return -1;
#else
int main(int argc, ORTCHAR_T * argv[]) {
#endif
	int ret = -1;
	try {
		// ret = real_main(argc, argv);
		ret = run_e4e(argc, argv);
	}
	catch (const std::exception& ex) {
		fprintf(stderr, "%s\n", ex.what());
	}
#ifdef _WIN32
	CoUninitialize();
#endif
	return ret;
}
