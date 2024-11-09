#pragma once

static char *CUSTOM_KERNEL = R"MPS_ROI_POOLING(
#include <metal_stdlib>
using namespace metal;

// Performs ROI Pooling
// Input: feature_map is a C x H x W tensor, rois are M x 5 (batch_index, x1, y1, x2, y2)
// Output: pooled_features is M x C x pool_height x pool_width
template<typename T>

kernel void roipooling_kernel(constant T* feature_map [[buffer(0)]],
                              constant T* rois        [[buffer(1)]],
                              device   T* pooled_features [[buffer(2)]],
                              constant int& pool_height [[buffer(3)]],
                              constant int& pool_width  [[buffer(4)]],
                              constant int& channels    [[buffer(5)]],
                              constant int& height      [[buffer(6)]],
                              constant int& width       [[buffer(7)]],
                              uint index [[thread_position_in_grid]]) {

    // Determine the ROI and feature map indices
    int roi_idx = index / (channels * pool_height * pool_width);  // Which ROI we are processing
    int c_idx = (index / (pool_height * pool_width)) % channels;  // Which channel
    int ph_idx = (index / pool_width) % pool_height;              // Which row in pool
    int pw_idx = index % pool_width;                              // Which column in pool

    // Extract ROI parameters
    int batch_idx = int(rois[roi_idx * 5]); // ROI's batch index
    float x1 = rois[roi_idx * 5 + 1] * width;
    float y1 = rois[roi_idx * 5 + 2] * height;
    float x2 = rois[roi_idx * 5 + 3] * width;
    float y2 = rois[roi_idx * 5 + 4] * height;

    // Ensure we clamp the boundaries of the ROI to the feature map size
    x1 = clamp(x1, 0.0, float(width - 1));
    y1 = clamp(y1, 0.0, float(height - 1));
    x2 = clamp(x2, 0.0, float(width - 1));
    y2 = clamp(y2, 0.0, float(height - 1));

    // Calculate the size of the ROI and bin size for pooling
    float roi_w = max(x2 - x1 + 1.0, 1.0); // Width of the ROI
    float roi_h = max(y2 - y1 + 1.0, 1.0); // Height of the ROI
    float bin_w = roi_w / float(pool_width);  // Width of each pooling bin
    float bin_h = roi_h / float(pool_height); // Height of each pooling bin

    // Determine the range of pixels for this pooling region
    float xstart = x1 + float(pw_idx) * bin_w;
    float ystart = y1 + float(ph_idx) * bin_h;
    float xend = x1 + float(pw_idx + 1) * bin_w;
    float yend = y1 + float(ph_idx + 1) * bin_h;

    // Clamp the pooling region within the feature map boundaries
    xstart = clamp(xstart, 0.0, float(width - 1));
    ystart = clamp(ystart, 0.0, float(height - 1));
    xend = clamp(xend, 0.0, float(width));
    yend = clamp(yend, 0.0, float(height));

    // Perform max pooling over the region defined by (xstart, xend) and (ystart, yend)
    float max_val = -INFINITY;
    for (float y = ystart; y < yend; y += 1.0) {
        for (float x = xstart; x < xend; x += 1.0) {
            int feature_map_index = batch_idx * channels * height * width +
                                    c_idx * height * width +
                                    int(y) * width + int(x);
            max_val = max(max_val, feature_map[feature_map_index]);
        }
    }

    // Store the result in the output pooled_features tensor
    pooled_features[index] = max_val;
}

// Template for half precision
template
[[host_name("roipooling_kernel_half")]]
kernel void roipooling_kernel<half>(constant half* feature_map [[buffer(0)]],
                                    constant half* rois        [[buffer(1)]],
                                    device   half* pooled_features [[buffer(2)]],
                                    constant int& pool_height [[buffer(3)]],
                                    constant int& pool_width  [[buffer(4)]],
                                    constant int& channels    [[buffer(5)]],
                                    constant int& height      [[buffer(6)]],
                                    constant int& width       [[buffer(7)]],
                                    uint index [[thread_position_in_grid]]);

// Template for float precision
template
[[host_name("roipooling_kernel_float")]]
kernel void roipooling_kernel<float>(constant float* feature_map [[buffer(0)]],
                                     constant float* rois        [[buffer(1)]],
                                     device   float* pooled_features [[buffer(2)]],
                                     constant int& pool_height [[buffer(3)]],
                                     constant int& pool_width  [[buffer(4)]],
                                     constant int& channels    [[buffer(5)]],
                                     constant int& height      [[buffer(6)]],
                                     constant int& width       [[buffer(7)]],
                                     uint index [[thread_position_in_grid]]);
)MPS_ROI_POOLING";
