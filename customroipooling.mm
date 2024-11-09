#include <torch/extension.h>
#include "customroipooling.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Helper function to retrieve the MTLBuffer from torch::Tensor.
void static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

torch::Tensor& dispatchROIPoolingKernel(const torch::Tensor& feature_map,
                                        const torch::Tensor& rois,
                                        torch::Tensor& pooled_features,
                                        int pool_height, int pool_width) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        int numThreads = pooled_features.numel();
        int channels = feature_map.size(1);
        int height = feature_map.size(2);
        int width = feature_map.size(3);

        // Load the custom ROI pooling shader.
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(customKernelLibrary, "Failed to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string kernel_name = "roipooling_kernel_" + 
                                   (feature_map.scalar_type() == torch::kFloat ? "float" : "half");

        id<MTLFunction> customROIPoolingFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(customROIPoolingFunction, "Failed to create function state object for ", kernel_name.c_str());

        id<MTLComputePipelineState> roipoolingPSO = [device newComputePipelineStateWithFunction:customROIPoolingFunction error:&error];
        TORCH_CHECK(roipoolingPSO, error.localizedDescription.UTF8String);

        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^{
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            [computeEncoder setComputePipelineState:roipoolingPSO];
            [computeEncoder setBuffer:getMTLBufferStorage(feature_map) offset:feature_map.storage_offset() * feature_map.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(rois) offset:rois.storage_offset() * rois.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(pooled_features) offset:pooled_features.storage_offset() * pooled_features.element_size() atIndex:2];
            [computeEncoder setBytes:&pool_height length:sizeof(int) atIndex:3];
            [computeEncoder setBytes:&pool_width length:sizeof(int) atIndex:4];
            [computeEncoder setBytes:&channels length:sizeof(int) atIndex:5];
            [computeEncoder setBytes:&height length:sizeof(int) atIndex:6];
            [computeEncoder setBytes:&width length:sizeof(int) atIndex:7];

            // Calculate grid size and thread group size.
            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
            NSUInteger threadGroupSize = roipoolingPSO.maxTotalThreadsPerThreadgroup;

            if (threadGroupSize > numThreads) {
                threadGroupSize = numThreads;
            }

            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            // Dispatch the compute command.
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [computeEncoder endEncoding];

            torch::mps::commit();
        });
    }
    return pooled_features;
}

// C++ function for dispatching the Metal ROI Pooling shader.
torch::Tensor mps_roipooling(const torch::Tensor &feature_map, const torch::Tensor &rois, int pool_height = 7, int pool_width = 7) {
    // Ensure input tensors are MPS tensors and contiguous
    TORCH_CHECK(feature_map.device().is_mps(), "feature_map must be a MPS tensor");
    TORCH_CHECK(feature_map.is_contiguous(), "feature_map must be contiguous");
    TORCH_CHECK(rois.device().is_mps(), "rois must be a MPS tensor");

    // Prepare the output tensor with appropriate size: [num_rois, channels, pool_height, pool_width]
    torch::Tensor pooled_features = torch::empty({rois.size(0), feature_map.size(1), pool_height, pool_width}, feature_map.options());

    // Dispatch the ROI pooling kernel
    dispatchROIPoolingKernel(feature_map, rois, pooled_features, pool_height, pool_width);

    // Return the pooled features
    return pooled_features;
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mps_roipooling", &mps_roipooling, "Custom Metal ROI Pooling",
          py::arg("feature_map"),
          py::arg("rois"),
          py::arg("pool_height") = 7,
          py::arg("pool_width") = 7);
}
