#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

void PrintMatrixf32(float* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

int main()
{
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
	printf("Found %d GPUs.\n", numGPUs);
    cudaSetDevice(0);
    
    int device;
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
	printf("Compute capability: %d.%d\n", devProp.major, devProp.minor);
    
	cudnnHandle_t cudnnHandle;
	cudnnCreate(&cudnnHandle);

	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);

	// simple 8 by 8 image with 3 channels, filter is 3 by 3 with 3 channels, stride is 1, padding is 0
	const uint32_t imageRows = 8;
	const uint32_t imageCols = 8;
	const uint32_t imageChannels = 1;
	const uint32_t filterRows = 3;
	const uint32_t filterCols = 3;
	const uint32_t filterChannels = 1;
	const uint32_t stride = 1;
	const uint32_t padding = 0;
	
	// calculate output size
	const uint32_t outputRows = 6;
	const uint32_t outputCols = 6;
	const uint32_t outputChannels = filterChannels;
	const uint32_t outputSize = outputRows * outputCols * outputChannels;
	
	// allocate memory for image, filter, and output
	float* image;
	float* filter;
	float* output;
	cudaMalloc(&image, imageRows * imageCols * imageChannels * sizeof(float));
	cudaMalloc(&filter, filterRows * filterCols * filterChannels * sizeof(float));
	cudaMalloc(&output, outputSize * sizeof(float));
	
	// initialize image and filter with random numbers
	curandGenerateUniform(curandGenerator, image, imageRows * imageCols * imageChannels);
	curandGenerateUniform(curandGenerator, filter, filterRows * filterCols * filterChannels);
	
	// create tensor descriptors
	cudnnTensorDescriptor_t imageTensorDesc;
	cudnnTensorDescriptor_t outputTensorDesc;
	cudnnCreateTensorDescriptor(&imageTensorDesc);
	cudnnCreateTensorDescriptor(&outputTensorDesc);
	cudnnSetTensor4dDescriptor(imageTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, imageChannels, imageRows, imageCols);
	cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outputChannels, outputRows, outputCols);
	
	// create filter descriptor
	cudnnFilterDescriptor_t filterTensorDesc;
	cudnnCreateFilterDescriptor(&filterTensorDesc);
	cudnnSetFilter4dDescriptor(filterTensorDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterChannels, filterRows, filterCols, 1);
	
	// set alpha and beta
	float alpha = 1.0f;
	float beta = 0.0f;
	
	// create convolution descriptor
	cudnnConvolutionDescriptor_t convDesc;
	cudnnCreateConvolutionDescriptor(&convDesc);
	cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
		
	// calculate workspace size
	size_t workspaceSize;
	cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, imageTensorDesc, filterTensorDesc, convDesc, outputTensorDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspaceSize);
	
	// allocate workspace
	void* workspace;
	cudaMalloc(&workspace, workspaceSize);
		
	// perform convolution
	cudnnConvolutionForward(cudnnHandle, &alpha, imageTensorDesc, image, filterTensorDesc, filter, convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, workspace, workspaceSize, &beta, outputTensorDesc, output);
	
	float* imageHost = new float[imageRows * imageCols * imageChannels];
	float* filterHost = new float[filterRows * filterCols * filterChannels];
	float* outputHost = new float[outputSize];
	cudaMemcpy(imageHost, image, imageRows * imageCols * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(filterHost, filter, filterRows * filterCols * filterChannels * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(outputHost, output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
	
	PrintMatrixf32(imageHost, imageRows, imageCols * imageChannels, "Image");
	PrintMatrixf32(filterHost, filterRows, filterCols * filterChannels, "Filter");
	PrintMatrixf32(outputHost, outputRows, outputCols * outputChannels, "Output");
	
	return 0;
}