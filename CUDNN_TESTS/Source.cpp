#include <cudnn.h>
#include <iostream>

int main() {
	cudnnHandle_t cudnnHandle;
	cudnnCreate(&cudnnHandle);
	
	const uint32_t BATCH_SIZE = 1;
	
	const uint32_t INPUT_CHANNELS = 1;
	const uint32_t INPUT_ROWS = 16;
	const uint32_t INPUT_COLS = 16;

	const uint32_t OUTPUT_CHANNELS = 1;
	const uint32_t OUTPUT_ROWS = 4;
	const uint32_t OUTPUT_COLS = 4;

	const uint32_t FILTER_ROWS = 4;
	const uint32_t FILTER_COLS = 4;

	const uint32_t PADDING = 0;
	const uint32_t STRIDE = 4;
	const uint32_t DILATION = 1;
	
	cudnnTensorDescriptor_t input_descriptor;
	cudnnTensorDescriptor_t output_descriptor;
	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnConvolutionDescriptor_t convolution_descriptor;
	
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnCreateTensorDescriptor(&output_descriptor);
	cudnnCreateFilterDescriptor(&kernel_descriptor);
	cudnnCreateConvolutionDescriptor(&convolution_descriptor);
	
	cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, BATCH_SIZE, INPUT_CHANNELS, INPUT_ROWS, INPUT_COLS);
	cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, BATCH_SIZE, OUTPUT_CHANNELS, OUTPUT_ROWS, OUTPUT_COLS);
	cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, OUTPUT_CHANNELS, INPUT_CHANNELS, FILTER_ROWS, FILTER_COLS);
	cudnnSetConvolution2dDescriptor(convolution_descriptor, PADDING, PADDING, STRIDE, STRIDE, DILATION, DILATION, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
	
	cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm;
	int maxPropagationAlgorithms;
	cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &maxPropagationAlgorithms);
	cudnnConvolutionFwdAlgoPerf_t* forwardPropagationAlgorithms = new cudnnConvolutionFwdAlgoPerf_t[maxPropagationAlgorithms];
	cudnnFindConvolutionForwardAlgorithm(cudnnHandle, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, maxPropagationAlgorithms, &maxPropagationAlgorithms, forwardPropagationAlgorithms);
	forwardPropagationAlgorithm = forwardPropagationAlgorithms[0].algo;
	delete[] forwardPropagationAlgorithms;
	printf("Forward propagation algorithm: %d\n\n", forwardPropagationAlgorithm);
	
	size_t workspaceBytes = 0;
	cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, forwardPropagationAlgorithm, &workspaceBytes);
	void* workspace;
	cudaMalloc(&workspace, workspaceBytes);
	
	float* gpuInput;
	float* gpuOutput;
	float* gpuFilter;
	
	cudaMalloc(&gpuInput, BATCH_SIZE * INPUT_CHANNELS * INPUT_ROWS * INPUT_COLS * sizeof(float));
	cudaMalloc(&gpuOutput, BATCH_SIZE * OUTPUT_CHANNELS * OUTPUT_ROWS * OUTPUT_COLS * sizeof(float));
	cudaMalloc(&gpuFilter, OUTPUT_CHANNELS * INPUT_CHANNELS * FILTER_ROWS * FILTER_COLS * sizeof(float));
	
	float* input = new float[BATCH_SIZE * INPUT_CHANNELS * INPUT_ROWS * INPUT_COLS];
	float* output = new float[BATCH_SIZE * OUTPUT_CHANNELS * OUTPUT_ROWS * OUTPUT_COLS];
	float* filter = new float[OUTPUT_CHANNELS * INPUT_CHANNELS * FILTER_ROWS * FILTER_COLS];

	for (int i = 0; i < BATCH_SIZE * INPUT_CHANNELS * INPUT_ROWS * INPUT_COLS; i++)
		input[i] = (float)rand() / (float)RAND_MAX;
	for (int i = 0; i < OUTPUT_CHANNELS * INPUT_CHANNELS * FILTER_ROWS * FILTER_COLS; i++)
		filter[i] = (float)rand() / (float)RAND_MAX;
	
	cudaMemcpy(gpuInput, input, BATCH_SIZE * INPUT_CHANNELS * INPUT_ROWS * INPUT_COLS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuFilter, filter, OUTPUT_CHANNELS * INPUT_CHANNELS * FILTER_ROWS * FILTER_COLS * sizeof(float), cudaMemcpyHostToDevice);
	
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cudnnConvolutionForward(cudnnHandle, &alpha, input_descriptor, gpuInput, kernel_descriptor, gpuFilter, convolution_descriptor, forwardPropagationAlgorithm, workspace, workspaceBytes, &beta, output_descriptor, gpuOutput);
	
	cudaMemcpy(output, gpuOutput, BATCH_SIZE * OUTPUT_CHANNELS * OUTPUT_ROWS * OUTPUT_COLS * sizeof(float), cudaMemcpyDeviceToHost);

	// print the input and filter
	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		for (uint32_t j = 0; j < INPUT_CHANNELS; j++)
		{
			for (uint32_t k = 0; k < INPUT_ROWS; k++)
			{
				for (uint32_t l = 0; l < INPUT_COLS; l++)
				{
					printf("%f ", input[i * INPUT_CHANNELS * INPUT_ROWS * INPUT_COLS + j * INPUT_ROWS * INPUT_COLS + k * INPUT_COLS + l]);
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}

	for (uint32_t i = 0; i < OUTPUT_CHANNELS; i++)
	{
		for (uint32_t j = 0; j < INPUT_CHANNELS; j++)
		{
			for (uint32_t k = 0; k < FILTER_ROWS; k++)
			{
				for (uint32_t l = 0; l < FILTER_COLS; l++)
				{
					printf("%f ", filter[i * INPUT_CHANNELS * FILTER_ROWS * FILTER_COLS + j * FILTER_ROWS * FILTER_COLS + k * FILTER_COLS + l]);
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");

	// printing the output
	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		for (uint32_t j = 0; j < OUTPUT_CHANNELS; j++)
		{
			for (uint32_t k = 0; k < OUTPUT_ROWS; k++)
			{
				for (uint32_t l = 0; l < OUTPUT_COLS; l++)
				{
					printf("%f ", output[i * OUTPUT_CHANNELS * OUTPUT_ROWS * OUTPUT_COLS + j * OUTPUT_ROWS * OUTPUT_COLS + k * OUTPUT_COLS + l]);
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}

	// cpu implementation
	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		for (uint32_t j = 0; j < OUTPUT_CHANNELS; j++)
		{
			for (uint32_t k = -PADDING; k < OUTPUT_ROWS * STRIDE - PADDING; k += STRIDE)
			{
				for (uint32_t l = -PADDING; l < OUTPUT_COLS * STRIDE - PADDING; l += STRIDE)
				{
					float sum = 0.0f;
					for (uint32_t m = 0; m < INPUT_CHANNELS; m++)
					{
						for (uint32_t n = 0; n < FILTER_ROWS; n++)
						{
							for (uint32_t o = 0; o < FILTER_COLS; o++)
							{
								sum += input[i * INPUT_CHANNELS * INPUT_ROWS * INPUT_COLS + m * INPUT_ROWS * INPUT_COLS + (k + n) * INPUT_COLS + (l + o)] * filter[j * INPUT_CHANNELS * FILTER_ROWS * FILTER_COLS + m * FILTER_ROWS * FILTER_COLS + n * FILTER_COLS + o];
							}
						}
					}
					printf("%f ", sum);
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}
}