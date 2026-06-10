#!/usr/bin/env python3
"""Fine-grained per-block timing profile of the Pixal3D teacher (1536_cascade, geometry+tex).

Re-implements Pixal3DImageTo3DPipeline.run() inline (mirroring it statement-for-statement)
with (a) CUDA-synced wall-clock timers per block and (b) CUDA-event timing of every flow-DiT
forward (so CFG 2-forward steps vs single-forward steps are measured, not inferred).

Fine blocks:
  image_load, preprocess, moge_camera,
  cond_ss, flow_ss, ss_decode, ss_support,
  cond_lr, flow_lr, upsample_quantize,
  cond_hr, flow_hr, cond_tex, flow_tex,
  decode_shape, decode_tex, mesh_post
Chunks:
  pre = image_load+preprocess | camera = moge_camera | cond = cond_* |
  flow = flow_* | support = ss_decode+ss_support+upsample_quantize |
  decode = decode_shape+decode_tex+mesh_post

Env/driver conventions copied from 3d_dmd's run_inference_pixal3d.py:
cwd MUST be the Pixal3D clone root; same conda env; config = std_full pixal3d.yaml.
Outputs: per-object JSONL + aggregate JSON + markdown summary printed to stdout.
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_PIPELINE_DIR = Path("/gs/fs/tga-koike-shanda2/sk/3d-distill/pipeline")
sys.path.insert(0, str(EVAL_PIPELINE_DIR))

from src.utils.inference_config import get_inference_params, load_and_filter_samples  # noqa: E402


def _unshadow_src():
    """torch.hub NAF repo needs to own the `src` module name (see run_inference_pixal3d.py)."""
    for _k in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
        del sys.modules[_k]
    if str(EVAL_PIPELINE_DIR) in sys.path:
        sys.path.remove(str(EVAL_PIPELINE_DIR))


class BlockTimer:
    def __init__(self):
        self.spans = {}

    @contextmanager
    def span(self, name):
        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.spans[name] = self.spans.get(name, 0.0) + (time.perf_counter() - t0)


class ForwardRecorder:
    """Wraps flow-DiT .forward with CUDA events; resolved per object after a sync."""

    def __init__(self):
        import torch
        self._torch = torch
        self.stage = None
        self.events = []

    def wrap(self, model):
        torch = self._torch
        orig = model.forward
        rec = self

        def timed_forward(*args, **kwargs):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out = orig(*args, **kwargs)
            e1.record()
            rec.events.append((rec.stage, e0, e1))
            return out

        model.forward = timed_forward

    def drain(self):
        self._torch.cuda.synchronize()
        per_stage = {}
        for stage, e0, e1 in self.events:
            per_stage.setdefault(stage, []).append(round(e0.elapsed_time(e1), 2))
        self.events = []
        return per_stage


CHUNK_OF = {
    "image_load": "pre", "preprocess": "pre", "moge_camera": "camera",
    "cond_ss": "cond", "cond_lr": "cond", "cond_hr": "cond", "cond_tex": "cond",
    "flow_ss": "flow", "flow_lr": "flow", "flow_hr": "flow", "flow_tex": "flow",
    "ss_decode": "support", "ss_support": "support", "upsample_quantize": "support",
    "decode_shape": "decode", "decode_tex": "decode", "mesh_post": "decode",
}
FINE_ORDER = ["image_load", "preprocess", "moge_camera", "cond_ss", "flow_ss", "ss_decode",
              "ss_support", "cond_lr", "flow_lr", "upsample_quantize", "cond_hr", "flow_hr",
              "cond_tex", "flow_tex", "decode_shape", "decode_tex", "mesh_post"]
CHUNK_ORDER = ["pre", "camera", "cond", "support", "flow", "decode"]
N_STEPS = 12  # all four samplers run 12 steps (pipeline.json)


def profile_one(pipeline, pixal_inf, moge, sample, seed, rec, tmp_dir):
    import torch
    from PIL import Image

    # run() is @torch.no_grad(); this inline mirror must be too, or 12-step sampling
    # accumulates autograd graphs and OOMs.
    with torch.no_grad():
        return _profile_one_inner(pipeline, pixal_inf, moge, sample, seed, rec, tmp_dir)


def _profile_one_inner(pipeline, pixal_inf, moge, sample, seed, rec, tmp_dir):
    import torch
    from PIL import Image

    timer = BlockTimer()
    torch.cuda.reset_peak_memory_stats()
    t_total0 = time.perf_counter()

    with timer.span("image_load"):
        image = Image.open(sample.input_image)
        image.load()
    # Mirror preprocess_image's actual rembg-skip condition: RGBA with a *real* alpha channel
    # (an all-255 alpha still pays the BiRefNet forward).
    has_alpha = image.mode == "RGBA" and image.getextrema()[3][0] < 255

    with timer.span("preprocess"):
        img_pre = pipeline.preprocess_image(image)

    tmp_path = os.path.join(tmp_dir, "_tmp_preprocessed.png")
    with timer.span("moge_camera"):
        img_pre.save(tmp_path)
        camera_params = pixal_inf.get_camera_params_wild_moge(
            tmp_path, moge, device="cuda",
            mesh_scale=1.0, extend_pixel=0, image_resolution=512)
        os.remove(tmp_path)

    camera_angle_x = camera_params["camera_angle_x"]
    distance = camera_params["distance"]
    mesh_scale = camera_params.get("mesh_scale", 1.0)
    torch.manual_seed(seed)

    # ---- Stage 1: Sparse Structure ----
    with timer.span("cond_ss"):
        cond_ss = pipeline.get_proj_cond_ss(
            [img_pre], camera_angle_x=camera_angle_x, distance=distance, mesh_scale=mesh_scale)
    flow_ss = pipeline.models["sparse_structure_flow_model"]
    reso = flow_ss.resolution
    rec.stage = "ss"
    with timer.span("flow_ss"):
        # Noise creation inside the span (RNG order unchanged: still right after cond_ss).
        noise = torch.randn(1, flow_ss.in_channels, reso, reso, reso).to(pipeline.device)
        z_s = pipeline.sparse_structure_sampler.sample(
            flow_ss, noise, **cond_ss, **pipeline.sparse_structure_sampler_params,
            verbose=False).samples
    rec.stage = None
    with timer.span("ss_decode"):
        decoded = pipeline.models["sparse_structure_decoder"](z_s) > 0
    with timer.span("ss_support"):
        ss_res = 32
        if ss_res != decoded.shape[2]:
            ratio = decoded.shape[2] // ss_res
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    n_lr = int(coords.shape[0])
    del cond_ss, z_s, decoded
    torch.cuda.empty_cache()

    # ---- Stage 2: Shape LR 512 ----
    with timer.span("cond_lr"):
        cond_lr = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_512, [img_pre], coords,
            camera_angle_x=camera_angle_x, distance=distance, mesh_scale=mesh_scale)
    rec.stage = "lr"
    with timer.span("flow_lr"):
        lr_slat = pipeline.sample_shape_slat(
            cond_lr, pipeline.models["shape_slat_flow_model_512"], coords)
    rec.stage = None
    del cond_lr
    torch.cuda.empty_cache()

    # ---- Stage 3a: Upsample LR -> HR (mirrors run() incl. token-cap loop) ----
    max_num_tokens = 49152
    with timer.span("upsample_quantize"):
        hr_coords = pipeline.models["shape_slat_decoder"].upsample(lr_slat, upsample_times=4)
        lr_resolution = 512
        actual_hr_resolution = 1536
        while True:
            grid_res = actual_hr_resolution // 16
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
            ], dim=1)
            hr_coords_unique = quant_coords.unique(dim=0)
            if hr_coords_unique.shape[0] < max_num_tokens or actual_hr_resolution == 1024:
                break
            actual_hr_resolution -= 128
    actual_grid_res = actual_hr_resolution // 16
    n_hr = int(hr_coords_unique.shape[0])
    del lr_slat, hr_coords, quant_coords
    torch.cuda.empty_cache()

    # ---- Stage 3b: Shape HR ----
    with timer.span("cond_hr"):
        cond_hr = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_1024, [img_pre], hr_coords_unique,
            camera_angle_x=camera_angle_x, distance=distance, mesh_scale=mesh_scale,
            grid_resolution_override=actual_grid_res)
    rec.stage = "hr"
    with timer.span("flow_hr"):
        shape_slat = pipeline.sample_shape_slat(
            cond_hr, pipeline.models["shape_slat_flow_model_1024"], hr_coords_unique)
    rec.stage = None
    del cond_hr, hr_coords_unique
    torch.cuda.empty_cache()

    # ---- Stage 4: Texture ----
    with timer.span("cond_tex"):
        cond_tex = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024, [img_pre], shape_slat.coords,
            camera_angle_x=camera_angle_x, distance=distance, mesh_scale=mesh_scale,
            grid_resolution_override=actual_grid_res)
    rec.stage = "tex"
    with timer.span("flow_tex"):
        tex_slat = pipeline.sample_tex_slat(
            cond_tex, pipeline.models["tex_slat_flow_model_1024"], shape_slat)
    rec.stage = None
    del cond_tex
    torch.cuda.empty_cache()

    # ---- Stage 5: Decode (decode_latent split into 3 blocks) ----
    # Note: mesh_post excludes decode_latent's MeshWithVoxel packing (tensor-handle
    # wrapping, ~0 cost) — fill_holes is the only real work in that loop.
    with timer.span("decode_shape"):
        with torch.no_grad():
            meshes, subs = pipeline.decode_shape_slat(shape_slat, actual_hr_resolution)
    with timer.span("decode_tex"):
        with torch.no_grad():
            tex_voxels = pipeline.decode_tex_slat(tex_slat, subs)
    with timer.span("mesh_post"):
        for m in meshes:
            m.fill_holes()
    verts = int(meshes[0].vertices.shape[0])
    faces = int(meshes[0].faces.shape[0])

    forwards = rec.drain()
    total = time.perf_counter() - t_total0

    entry = {
        "object_id": sample.object_id,
        "category": getattr(sample, "category", None),
        "has_alpha": has_alpha,
        "n_lr_tokens": n_lr,
        "n_hr_tokens": n_hr,
        "hr_resolution": actual_hr_resolution,
        "verts": verts,
        "faces": faces,
        "spans_s": {k: round(v, 4) for k, v in timer.spans.items()},
        "total_s": round(total, 3),
        "forward_ms": forwards,  # per stage: list of per-forward ms in call order
        "n_forwards": {k: len(v) for k, v in forwards.items()},
        "peak_gpu_gb": round(__import__("torch").cuda.max_memory_allocated() / 1e9, 2),
    }
    del shape_slat, tex_slat, meshes, subs, tex_voxels
    __import__("torch").cuda.empty_cache()
    return entry


def aggregate(entries):
    import statistics as st
    agg = {"n_objects": len(entries), "fine": {}, "chunks": {}, "flow_stages": {}}
    mean_total = st.mean(e["total_s"] for e in entries)
    for blk in FINE_ORDER:
        vals = [e["spans_s"].get(blk, 0.0) for e in entries]
        agg["fine"][blk] = {
            "mean_s": round(st.mean(vals), 3), "median_s": round(st.median(vals), 3),
            "std_s": round(st.pstdev(vals), 3), "min_s": round(min(vals), 3),
            "max_s": round(max(vals), 3), "share_pct": round(100 * st.mean(vals) / mean_total, 1),
        }
    for ck in CHUNK_ORDER:
        vals = [sum(e["spans_s"].get(b, 0.0) for b, c in CHUNK_OF.items() if c == ck)
                for e in entries]
        agg["chunks"][ck] = {"mean_s": round(st.mean(vals), 3),
                             "share_pct": round(100 * st.mean(vals) / mean_total, 1)}
    for stage in ["ss", "lr", "hr", "tex"]:
        fwd_lists = [e["forward_ms"].get(stage, []) for e in entries]
        n_fwd = [len(f) for f in fwd_lists]
        cfg_steps = [max(0, n - N_STEPS) for n in n_fwd]
        all_ms = [ms for f in fwd_lists for ms in f]
        agg["flow_stages"][stage] = {
            "mean_forwards": round(st.mean(n_fwd), 1),
            "mean_cfg_steps": round(st.mean(cfg_steps), 1),
            "mean_ms_per_forward": round(st.mean(all_ms), 1) if all_ms else None,
            "mean_total_ms": round(st.mean([sum(f) for f in fwd_lists]), 1),
        }
    agg["mean_total_s"] = round(mean_total, 2)
    agg["tokens"] = {
        "n_lr_mean": round(st.mean(e["n_lr_tokens"] for e in entries)),
        "n_hr_mean": round(st.mean(e["n_hr_tokens"] for e in entries)),
        "n_hr_min": min(e["n_hr_tokens"] for e in entries),
        "n_hr_max": max(e["n_hr_tokens"] for e in entries),
        "hr_res_counts": {str(r): sum(1 for e in entries if e["hr_resolution"] == r)
                          for r in sorted({e["hr_resolution"] for e in entries})},
    }
    agg["peak_gpu_gb_max"] = max(e["peak_gpu_gb"] for e in entries)
    return agg


def print_markdown(agg):
    print("\n## Pixal3D teacher profile (n=%d, mean total %.2fs)" % (agg["n_objects"], agg["mean_total_s"]))
    print("\n### Fine blocks\n")
    print("| block | mean s | median | std | min | max | share % |")
    print("|---|---|---|---|---|---|---|")
    for blk in FINE_ORDER:
        f = agg["fine"][blk]
        print(f"| {blk} | {f['mean_s']} | {f['median_s']} | {f['std_s']} | {f['min_s']} | {f['max_s']} | {f['share_pct']} |")
    print("\n### Chunks\n")
    print("| chunk | mean s | share % |")
    print("|---|---|---|")
    for ck in CHUNK_ORDER:
        c = agg["chunks"][ck]
        print(f"| {ck} | {c['mean_s']} | {c['share_pct']} |")
    timed = sum(agg["fine"][b]["mean_s"] for b in FINE_ORDER)
    resid = agg["mean_total_s"] - timed
    print(f"| (untimed residual: noise gen, empty_cache, event drain) | {resid:.3f} | "
          f"{100 * resid / agg['mean_total_s']:.1f} |")
    print("\n### Flow stages (per-forward CUDA events)\n")
    print("| stage | forwards | CFG steps | ms/forward | total ms |")
    print("|---|---|---|---|---|")
    for stage in ["ss", "lr", "hr", "tex"]:
        s = agg["flow_stages"][stage]
        print(f"| {stage} | {s['mean_forwards']} | {s['mean_cfg_steps']} | {s['mean_ms_per_forward']} | {s['mean_total_ms']} |")
    print("\n### Tokens / memory\n")
    print(json.dumps(agg["tokens"], indent=2))
    print(f"peak_gpu_gb_max: {agg['peak_gpu_gb_max']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/gs/bs/tga-koike-shanda/3d_dmd_eval/std_full/configs/pixal3d.yaml")
    parser.add_argument("--num-objects", type=int, default=66)
    parser.add_argument("--warmup", type=int, default=2,
                        help="first N objects recorded but excluded from aggregates")
    parser.add_argument("--out-dir", default="/gs/fs/tga-koike-shanda2/sk/scratch/pixal3d_profile")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    samples = load_and_filter_samples(cfg, max_samples_override=None, manifest_key="test_manifest")
    inf_params = get_inference_params(cfg, "pixal3d")
    model_path = inf_params.get("model_path", "pretrained/Pixal3D")
    pipeline_type = inf_params.get("pipeline_type", "1536_cascade")
    assert pipeline_type == "1536_cascade", \
        f"profile mirrors the 1536_cascade path only, config says {pipeline_type}"

    # Evenly-spaced subsample over the manifest for category coverage.
    import numpy as np
    n_manifest = len(samples)
    idx = np.unique(np.linspace(0, len(samples) - 1, args.num_objects).astype(int))
    samples = [samples[i] for i in idx]
    print(f"Profiling {len(samples)} objects (warmup {args.warmup}) from manifest of {n_manifest}")

    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _unshadow_src()
    import inference as pixal_inf  # cwd = Pixal3D clone root
    pipeline = pixal_inf.init_pipeline(model_path, low_vram=False)
    print("[MoGe-2] Loading camera-estimation model...")
    moge = pixal_inf.load_moge_model(device="cuda")

    rec = ForwardRecorder()
    for key in ["sparse_structure_flow_model", "shape_slat_flow_model_512",
                "shape_slat_flow_model_1024", "tex_slat_flow_model_1024"]:
        rec.wrap(pipeline.models[key])

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "profile.jsonl")
    entries = []
    n_failed = 0
    with open(jsonl_path, "w") as jf:
        for i, s in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] {s.object_id}", flush=True)
            try:
                e = profile_one(pipeline, pixal_inf, moge, s, args.seed, rec, args.out_dir)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                rec.events = []
                rec.stage = None
                tmp = os.path.join(args.out_dir, "_tmp_preprocessed.png")
                if os.path.exists(tmp):
                    os.remove(tmp)
                n_failed += 1
                jf.write(json.dumps({"object_id": s.object_id, "status": "failed",
                                     "error": str(ex)}) + "\n")
                jf.flush()
                print(f"  FAILED: {ex}", flush=True)
                continue
            e["warmup"] = i < args.warmup
            jf.write(json.dumps(e) + "\n")
            jf.flush()
            print(f"  total={e['total_s']}s n_hr={e['n_hr_tokens']} hr_res={e['hr_resolution']} "
                  f"peak={e['peak_gpu_gb']}GB", flush=True)
            if not e["warmup"]:
                entries.append(e)

    if entries:
        agg = aggregate(entries)
        agg["n_failed"] = n_failed
        with open(os.path.join(args.out_dir, "aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2)
        print_markdown(agg)
    print(f"DONE (n_failed={n_failed})")
    if n_failed > len(samples) // 4:
        sys.exit(2)


if __name__ == "__main__":
    main()
