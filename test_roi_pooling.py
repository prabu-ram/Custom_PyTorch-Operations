import os
import time
import torch
from compiler import *
from roi_pooling import *
from torchvision.ops import roi_pool

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

#  MPS device
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    raise RuntimeError("MPS device is not available. Make sure you are using a compatible macOS version and PyTorch build.")

def test_speedup():
    custom_mps_roipooling = 0
    default_roipooling = 0
    
    feature_map = torch.randn(1, 3, 32, 32, dtype=torch.float32, device=mps_device)
    rois = torch.tensor([[0, 0, 0, 16, 16]], dtype=torch.float32, device=mps_device)

    print("Feature Map Type:", feature_map.dtype)
    print("ROIs Type:", rois.dtype)
    
    custom_model = CustomMPSROIPoolingModel().to(mps_device)
    torchvision_model = TorchvisionROIPoolingModel().to(mps_device)

    # time
    for _ in range(100):
        start = time.time()
        torchvision_model.forward(feature_map, rois)
        torch.mps.synchronize()
        default_roipooling += time.time() - start

        start = time.time()
        custom_model.forward(feature_map, rois)
        torch.mps.synchronize()
        custom_mps_roipooling += time.time() - start

    speedup = default_roipooling / custom_mps_roipooling
    print('Default ROI Pooling: {:.3f} us | Custom Kernel MPS ROI Pooling {:.3f} us ({:.3f} times faster)'.format(
        default_roipooling * 1e6 / 100, custom_mps_roipooling * 1e6 / 100, speedup))

# Tests the correctness of the custom ROI Pooling kernel.
def test_correctness():
    custom_roipooling = MPSROIPooling().to(mps_device)
    feature_map = torch.randn(1, 3, 32, 32, dtype=torch.float32, device=mps_device)  # Should be float32
    rois = torch.tensor([[0, 0, 0, 16, 16]], dtype=torch.float32, device=mps_device)

    print("Feature Map Type:", feature_map.dtype)
    print("ROIs Type:", rois.dtype)

    default_roipooling_op = roi_pool(feature_map, rois, output_size=(7, 7))
    custom_roipooling_op = custom_roipooling(feature_map, rois)

    # compare results
    print("Default ROI Pooling Output:\n", default_roipooling_op)
    print("Custom ROI Pooling Output:\n", custom_roipooling_op)

   
    if custom_roipooling_op.numel() > 0: 
        print("Debug pooled features (first few elements):", custom_roipooling_op[0, 0, 0, :4])

  
    torch.testing.assert_close(custom_roipooling_op, default_roipooling_op)

    torch.testing.assert_close(custom_roipooling_op, default_roipooling_op)


# correctness and speedup.
def test_roi_pooling():
    test_correctness()
    test_speedup()

if __name__ == "__main__":
    test_roi_pooling()
