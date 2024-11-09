import torch.utils.cpp_extension

compiled_lib = torch.utils.cpp_extension.load(
    name='CustomROIPooling',
    sources=['CustomROIPooling.mm'],  # Your custom Objective-C++ file for ROI Pooling
    extra_cflags=['-std=c++17'],
)
