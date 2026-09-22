"""Exercise GPU extension kernels, not just imports. Does not need model weights."""
import sys
from pathlib import Path

print('Importing NumPy...', flush=True)
import numpy as np
print('Importing PyTorch...', flush=True)
import torch
print('Importing NVDiffRast...', flush=True)
import nvdiffrast.torch as dr
print('Importing PyTorch3D...', flush=True)
from pytorch3d.ops import knn_points
print('Importing Warp...', flush=True)
import warp as wp
print('All imports succeeded.', flush=True)


@wp.kernel
def double_values(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i] * 2.0


def main():
    print('Checking PyTorch version and CUDA device...', flush=True)
    assert torch.version.cuda == '13.0', torch.version.cuda
    assert torch.__version__.startswith('2.9.1'), torch.__version__
    assert torch.cuda.is_available(), 'CUDA device unavailable'
    print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0))
    print('Running PyTorch CUDA matrix multiplication...', flush=True)
    x = torch.eye(16, device='cuda')
    torch.testing.assert_close(x @ x, x)

    print('Running PyTorch3D CUDA KNN...', flush=True)
    points = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], device='cuda')
    nearest = knn_points(points, points, K=1)
    torch.testing.assert_close(nearest.dists, torch.zeros_like(nearest.dists))
    assert nearest.idx[0, :, 0].tolist() == [0, 1]
    print('PyTorch3D CUDA KNN: OK')

    print('Running NVDiffRast CUDA rasterization...', flush=True)
    ctx = dr.RasterizeCudaContext()
    vertices = torch.tensor([[[-.8, -.8, 0., 1.], [.8, -.8, 0., 1.], [0., .8, 0., 1.]]],
                            dtype=torch.float32, device='cuda')
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')
    raster, _ = dr.rasterize(ctx, vertices, triangles, resolution=[32, 32])
    assert (raster[..., 3] > 0).any().item(), 'Rasterizer produced an empty image'
    assert torch.isfinite(raster).all().item()
    torch.cuda.synchronize()
    print('NVDiffRast CUDA rasterization: OK')

    print('Running Warp CUDA kernel...', flush=True)
    wp.init()
    values = wp.array(np.array([1., 2., 3.], dtype=np.float32), device='cuda:0')
    wp.launch(double_values, dim=3, inputs=[values], device='cuda:0')
    wp.synchronize()
    np.testing.assert_allclose(values.numpy(), [2., 4., 6.])
    print('Warp CUDA kernel: OK')

    print('Running mycpp pose clustering...', flush=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'mycpp' / 'build'))
    import mycpp
    pose = np.eye(4, dtype=np.float32)
    # Match estimater.py: arrays shaped (N, 4, 4), not Python lists.
    clustered = mycpp.cluster_poses(30., .01, np.stack([pose, pose]), pose[None])
    assert len(clustered) == 1
    print('mycpp pose clustering: OK')
    print('CUDA 13.0 GPU smoke checks passed; full FoundationPose inference is not tested here.')


if __name__ == '__main__':
    main()
