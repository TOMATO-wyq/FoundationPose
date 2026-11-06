#!/usr/bin/env python3
"""Real SAM smoke test, including FP's global CUDA tensor default and offload."""
import argparse
from pathlib import Path
import json
import time

from evaluate_kfs_masks import read_frame
from kfs_segmentation import KFSSegmenter, OfficialSAM, SegmentationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--frame", default="000000.png")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    import torch
    sam = OfficialSAM(args.checkpoint, "configs/sam2.1/sam2.1_hiera_s.yaml", "cuda")
    segmenter = KFSSegmenter(SegmentationConfig().validate(), sam, refinement=True)
    rgb, depth = read_frame(args.sequence, args.frame, .5)
    cold_allocated = torch.cuda.memory_allocated()
    # GEMM/attention libraries may retain framework workspaces after first use.
    # Test model offload separately, then test repeated cycles against a warmed baseline.
    segmenter.segment(rgb, depth)
    baseline = torch.cuda.memory_allocated()
    original_type = torch.tensor(0.).type()
    records = []
    try:
        for emulate in (False, True):
            if emulate:
                # Original FP refiner sets this globally; SAM recovery must survive it.
                torch.set_default_tensor_type("torch.cuda.FloatTensor")
            start = time.perf_counter()
            result = segmenter.segment(rgb, depth)
            allocated = torch.cuda.memory_allocated()
            assert not sam.frame_active
            assert all(parameter.device.type == "cpu" for parameter in sam.predictor.model.parameters())
            assert all(buffer.device.type == "cpu" for buffer in sam.predictor.model.buffers())
            for module in sam.predictor.model.modules():
                cache = getattr(module, "cache", None)
                if isinstance(cache, dict):
                    assert not any(torch.is_tensor(v) and v.is_cuda for v in cache.values())
                frequencies = getattr(module, "freqs_cis", None)
                assert not (torch.is_tensor(frequencies) and frequencies.is_cuda)
            if allocated > baseline + 1024*1024:
                import gc
                import warnings
                tensors = []
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for value in gc.get_objects():
                        if isinstance(value, torch.Tensor) and value.is_cuda:
                            tensors.append({"shape": list(value.shape), "dtype": str(value.dtype),
                                            "bytes": value.numel()*value.element_size()})
                args.output.with_suffix(".failure.json").write_text(json.dumps({"allocated": allocated,
                    "baseline": baseline, "cuda_tensors": tensors}, indent=2))
                raise AssertionError(f"SAM retains CUDA allocations: {allocated-baseline} bytes")
            records.append({"fp_cuda_default": emulate, "mask_pixels": int(result.mask.sum()),
                            "total_s": time.perf_counter()-start, "timings": result.timings,
                            "post_offload_allocated_bytes": allocated})
    finally:
        torch.set_default_tensor_type(original_type)
        sam.end()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sam": sam.metadata, "cold_allocated_bytes": cold_allocated,
                                     "warmed_baseline_allocated_bytes": baseline,
                                     "allocator_note": "framework workspace allocations can persist; SAM parameters, buffers and caches must be CPU",
                                     "checks": records}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
