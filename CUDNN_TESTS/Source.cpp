#include <iostream>
#include <cuda_runtime.h>
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

	// simple matrix multiplication
	const uint32_t INPUTS = 8;
	const uint32_t OUTPUTS = 4;
	const uint32_t BATCH = 1;

	// create the input matrix
	cudnnTensorDescriptor_t inputDesc;
	cudnnCreateTensorDescriptor(&inputDesc);
	cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH, INPUTS, 1, 1);
	float* input;
	cudaMalloc(&input, BATCH * INPUTS * sizeof(float));
	curandGenerateUniform(curandGenerator, input, BATCH * INPUTS);

	// create the output matrix
	cudnnTensorDescriptor_t outputDesc;
	cudnnCreateTensorDescriptor(&outputDesc);
	cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH, OUTPUTS, 1, 1);
	float* output;
	cudaMalloc(&output, BATCH * OUTPUTS * sizeof(float));
	curandGenerateUniform(curandGenerator, output, BATCH * OUTPUTS);
	
	cudnnFilterDescriptor_t weightDesc;
	cudnnCreateFilterDescriptor(&weightDesc);
	cudnnSetFilter4dDescriptor(weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OUTPUTS, INPUTS, 1, 1);
	float* weight;
	cudaMalloc(&weight, OUTPUTS * INPUTS * sizeof(float));
	curandGenerateUniform(curandGenerator, weight, OUTPUTS * INPUTS);
	
	cudnnConvolutionDescriptor_t convDesc;
	cudnnCreateConvolutionDescriptor(&convDesc);
	cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
	
	cudnnConvolutionFwdAlgo_t algo;
	cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDesc, weightDesc, convDesc, outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
	
	size_t workspaceSize;
	cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc, weightDesc, convDesc, outputDesc, algo, &workspaceSize);
	void* workspace;
	cudaMalloc(&workspace, workspaceSize);
	
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cudnnConvolutionForward(cudnnHandle, &alpha, inputDesc, input, weightDesc, weight, convDesc, algo, workspace, workspaceSize, &beta, outputDesc, output);
	
	float* outputHost = new float[BATCH * OUTPUTS];
	cudaMemcpy(outputHost, output, BATCH * OUTPUTS * sizeof(float), cudaMemcpyDeviceToHost);
	
	PrintMatrixf32(outputHost, BATCH, OUTPUTS, "Output");
	
	return 0;
}