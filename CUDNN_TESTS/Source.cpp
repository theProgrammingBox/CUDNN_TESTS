#include <cudnn.h>
#include <iostream>

int main() {
	cudnnHandle_t cudnnHandle;
	cudnnCreate(&cudnnHandle);

	const uint32_t batchSize = 1;
	const uint32_t inputChannels = 1;		// number of images stacked
	const uint32_t inputImageRows = 4;		// height of the input image
	const uint32_t inputImageCols = 4;		// width of the input image
	const uint32_t outputChannels = 1;
	const uint32_t outputImageRows = 2;
	const uint32_t outputImageCols = 2;
	const uint32_t filterRows = 3;			// weight
	const uint32_t filterCols = 3;
	const uint32_t verticalStride = 1;		// how many pixels to move the filter down
	const uint32_t horizontalStride = 1;	// how many pixels to move the filter right
	const uint32_t verticalPadding = 0;
	const uint32_t horizontalPadding = 0;
	const uint32_t verticalDilation = 1;	// how many pixels to skip when convolving
	const uint32_t horizontalDilation = 1;	// how many pixels to skip when convolving
	
	cudnnTensorDescriptor_t input_descriptor;
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batchSize,
		/*channels=*/inputChannels,
		/*image_height=*/inputImageRows,
		/*image_width=*/inputImageCols);

	cudnnTensorDescriptor_t output_descriptor;
	cudnnCreateTensorDescriptor(&output_descriptor);
	cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batchSize,
		/*channels=*/outputChannels,
		/*image_height=*/outputImageRows,
		/*image_width=*/outputImageCols);

	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnCreateFilterDescriptor(&kernel_descriptor);
	cudnnSetFilter4dDescriptor(kernel_descriptor,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*out_channels=*/outputChannels,
		/*in_channels=*/inputChannels,
		/*kernel_height=*/filterRows,
		/*kernel_width=*/filterCols);

	cudnnConvolutionDescriptor_t convolution_descriptor;
	cudnnCreateConvolutionDescriptor(&convolution_descriptor);
	cudnnSetConvolution2dDescriptor(convolution_descriptor,
		/*pad_height=*/verticalPadding,
		/*pad_width=*/horizontalPadding,
		/*vertical_stride=*/verticalStride,
		/*horizontal_stride=*/horizontalStride,
		/*dilation_height=*/verticalDilation,
		/*dilation_width=*/horizontalDilation,
		/*mode=*/CUDNN_CROSS_CORRELATION,
		/*computeType=*/CUDNN_DATA_FLOAT);

	int maxPropagationAlgorithms;
	cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &maxPropagationAlgorithms);
	cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm;
	cudnnConvolutionFwdAlgoPerf_t* forwardPropagationAlgorithms = new cudnnConvolutionFwdAlgoPerf_t[maxPropagationAlgorithms];
	cudnnFindConvolutionForwardAlgorithm(cudnnHandle,
		input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		maxPropagationAlgorithms,
		&maxPropagationAlgorithms,
		forwardPropagationAlgorithms);
	forwardPropagationAlgorithm = forwardPropagationAlgorithms[0].algo;
	delete[] forwardPropagationAlgorithms;
	
	size_t workspaceBytes = 0;
	void* workspace = nullptr;
	cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		forwardPropagationAlgorithm,
		&workspaceBytes);
	cudaMalloc(&workspace, workspaceBytes);

	// now we can run the convolution
	const float alpha = 1.0f;
	const float beta = 0.0f;

	// creating the random input, output, and filter data
	float* input = new float[batchSize * inputChannels * inputImageRows * inputImageCols];
	float* output = new float[batchSize * outputChannels * outputImageRows * outputImageCols];
	float* filter = new float[outputChannels * inputChannels * filterRows * filterCols];
	input[0] = 0.0f;
	input[1] = 1.0f;
	input[2] = 0.0f;
	input[3] = 0.0f;
	
	input[4] = 0.0f;
	input[5] = 1.0f;
	input[6] = 0.0f;
	input[7] = 0.0f;

	input[8] = 0.0f;
	input[9] = 0.0f;
	input[10] = 1.0f;
	input[11] = 0.0f;
	
	input[12] = 0.0f;
	input[13] = 0.0f;
	input[14] = 0.0f;
	input[15] = 1.0f;

	filter[0] = 0.0f;
	filter[1] = 1.0f;
	filter[2] = 0.0f;
	
	filter[3] = 0.0f;
	filter[4] = 1.0f;
	filter[5] = 0.0f;
	
	filter[6] = 0.0f;
	filter[7] = 0.0f;
	filter[8] = 1.0f;

	// allocating memory on the gpu
	float* gpuInput;
	float* gpuOutput;
	float* gpuFilter;
	cudaMalloc(&gpuInput, batchSize * inputChannels * inputImageRows * inputImageCols * sizeof(float));
	cudaMalloc(&gpuOutput, batchSize * outputChannels * outputImageRows * outputImageCols * sizeof(float));
	cudaMalloc(&gpuFilter, outputChannels * inputChannels * filterRows * filterCols * sizeof(float));
	cudaMemcpy(gpuInput, input, batchSize * inputChannels * inputImageRows * inputImageCols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuFilter, filter, outputChannels * inputChannels * filterRows * filterCols * sizeof(float), cudaMemcpyHostToDevice);

	// running the convolution
	cudnnConvolutionForward(cudnnHandle,
		&alpha,
		input_descriptor,
		gpuInput,
		kernel_descriptor,
		gpuFilter,
		convolution_descriptor,
		forwardPropagationAlgorithm,
		workspace,
		workspaceBytes,
		&beta,
		output_descriptor,
		gpuOutput);

	// copying the output back to the cpu
	cudaMemcpy(output, gpuOutput, batchSize * outputChannels * outputImageRows * outputImageCols * sizeof(float), cudaMemcpyDeviceToHost);

	printf("verticalPadding: %d\n", verticalPadding);
	printf("horizontalPadding: %d\n", horizontalPadding);
	printf("verticalStride: %d\n", verticalStride);
	printf("horizontalStride: %d\n", horizontalStride);

	// printing the output
	for (uint32_t i = 0; i < batchSize; i++)
	{
		for (uint32_t j = 0; j < outputChannels; j++)
		{
			for (uint32_t k = 0; k < outputImageRows; k++)
			{
				for (uint32_t l = 0; l < outputImageCols; l++)
				{
					printf("%f ", output[i * outputChannels * outputImageRows * outputImageCols + j * outputImageRows * outputImageCols + k * outputImageCols + l]);
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}

	// cpu implementation
	for (uint32_t i = 0; i < batchSize; i++)
	{
		for (uint32_t j = 0; j < outputChannels; j++)
		{
			for (uint32_t k = -verticalPadding; k < outputImageRows * verticalStride - verticalPadding; k += verticalStride)
			{
				for (uint32_t l = -horizontalPadding; l < outputImageCols * horizontalStride - horizontalPadding; l += horizontalStride)
				{
					float sum = 0.0f;
					for (uint32_t m = 0; m < inputChannels; m++)
					{
						for (uint32_t n = 0; n < filterRows; n++)
						{
							for (uint32_t o = 0; o < filterCols; o++)
							{
								if (k + n >= 0 && k + n < inputImageRows && l + o >= 0 && l + o < inputImageCols)
								{
									sum += input[i * inputChannels * inputImageRows * inputImageCols + m * inputImageRows * inputImageCols + (k + n) * inputImageCols + (l + o)] * filter[j * inputChannels * filterRows * filterCols + m * filterRows * filterCols + n * filterCols + o];
								}
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