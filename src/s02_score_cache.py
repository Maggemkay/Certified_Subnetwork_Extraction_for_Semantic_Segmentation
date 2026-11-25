"""s02 · Hamblin score cache (ActGrad saliency, cached per image as in Anani App. D).

For every class c and every image x in D_c:
  - one forward pass records the activations a of all hook points (4,352 channels);
  - for every target f in CACHED_TARGETS, one backward pass gives ∂f/∂a, and per channel
        actgrad = mean_{h,w} |a · ∂f/∂a|        (Hamblin Eq. 5, filter level)
  - independent of the target, per channel
        activation = mean_{h,w} a               (Anani's "activation" scorer)

Targets: the class-c logit of the output, averaged over a pixel region
    pred  pixels the full model predicts as c
    gt    GT pixels of c
    all   every pixel
An empty region gives f = nan and all-zero actgrad scores for that image. Such images are
kept, but each one is named in the log (e.g. the model predicts no class-c pixel: weak discovery data).

Output: <discovery_dir>/s02_scores/class_XX.pt and run_info.json.
"""

import time

import numpy as np
import torch

import config as C
from common import datasets, model as M, utils
from common.datasets import VOC_CLASSES

USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "DATA_DIR", "CLASSES", "MODEL_NAME", "DEVICE",
    "DISCOVERY_DATASET", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "REQUIRE_PRED_IN_DISCOVERY", "DISCOVERY_AND_EVAL_SEED",
    "CACHED_TARGETS", "SMOKE_TEST",
]


def score_image(model, image, mask, c, device):
    """ActGrad scores [NUM_COMPONENTS] per target, activation means, f values and region sizes."""
    with M.ActivationRecorder(model) as recorder:
        logits = model(M.preprocess(image, device))["out"][0]  # [21, H, W]
    acts = [recorder.acts[name] for name in M.HOOK_POINTS]    # each [1, C, h, w]

    regions = {
        "pred": logits.argmax(dim=0) == c,
        "gt": torch.as_tensor(mask == c, device=device),
        "all": torch.ones_like(logits[0], dtype=torch.bool),
    }
    scores, f_values, region_pixels = {}, {}, {}
    for target in C.CACHED_TARGETS:
        region = regions[target]
        region_pixels[target] = int(region.sum())
        if region_pixels[target] == 0:
            scores[target] = torch.zeros(M.NUM_COMPONENTS)
            f_values[target] = float("nan")
            continue
        f = logits[c][region].mean()
        grads = torch.autograd.grad(f, acts, retain_graph=True)
        with torch.no_grad():  # the scores must not keep the autograd graph alive
            scores[target] = torch.cat([(a * g).abs().mean(dim=(2, 3))[0] for a, g in zip(acts, grads)]).cpu()
        f_values[target] = f.item()

    act_mean = torch.cat([a.detach().mean(dim=(2, 3))[0] for a in acts]).cpu()
    return scores, act_mean, f_values, region_pixels


def score_class(c, entry, discovery, model, device):
    per_image = []
    for i, image_id in zip(entry["indices"], entry["ids"]):
        assert discovery.ids[i] == image_id, f"dataset order changed: {discovery.ids[i]} != {image_id}"
        image, mask = discovery[i]
        per_image.append(score_image(model, image, mask, c, device))

    scores, act_mean, f_values, region_pixels = zip(*per_image)
    return {
        "class": c,
        "name": VOC_CLASSES[c],
        "ids": entry["ids"],
        "indices": entry["indices"],
        "scores": {t: torch.stack([s[t] for s in scores]) for t in C.CACHED_TARGETS},  # [|D|, 4352]
        "act_mean": torch.stack(act_mean),                                              # [|D|, 4352]
        "f_values": {t: torch.tensor([f[t] for f in f_values]) for t in C.CACHED_TARGETS},
        "region_pixels": {t: torch.tensor([r[t] for r in region_pixels]) for t in C.CACHED_TARGETS},
        "layer_slices": {name: (s.start, s.stop) for name, s in M.LAYER_SLICES.items()},
    }


def layer_summary(results, target):
    """Per layer, averaged over classes: share of the summed score mass, % channels with a
    zero sum over D_c, % zero channels per single image, and the mass held by the top 10 %."""
    rows = {name: [] for name in M.HOOK_POINTS}
    for result in results:
        per_image = result["scores"][target].numpy()     # [|D|, 4352]
        total = per_image.sum(axis=0)                    # cumulative saliency (Hamblin Eq. 3)
        for name, s in M.LAYER_SLICES.items():
            layer_total = np.sort(total[s])[::-1]
            top = max(1, round(0.1 * len(layer_total)))
            rows[name].append([
                layer_total.sum() / total.sum(),
                np.mean(layer_total == 0),
                np.mean(per_image[:, s] == 0),
                layer_total[:top].sum() / layer_total.sum() if layer_total.sum() > 0 else np.nan,
            ])
    print(f"\nScore distribution per layer, target '{target}' (mean over classes):")
    print(f"{'layer':8s} {'mass share':>11s} {'zero sum':>9s} {'zero/img':>9s} {'top-10% mass':>13s}")
    for name, values in rows.items():
        share, zero_sum, zero_img, top_mass = np.nanmean(values, axis=0)
        print(f"{name:8s} {share:11.3f} {zero_sum:8.1%} {zero_img:8.1%} {top_mass:13.3f}")


def main():
    utils.start_run("s02_score_cache", USED_PARAMS)
    discovery_and_eval_lists = utils.load_discovery_and_eval_lists()
    out_dir = utils.score_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = utils.get_device()
    model = M.load_model(C.MODEL_NAME, device)
    discovery = datasets.get_dataset(C.DISCOVERY_DATASET)

    print("\nPer class and target: mean f over D_c (images with an empty target region).")
    print(f"{'':3s} {'class':12s} {'|D_c|':>5s}" + "".join(f" {'f_' + t:>14s}" for t in C.CACHED_TARGETS)
          + f" {'time':>6s}")
    results = []
    for c in C.CLASSES:
        start = time.time()
        result = score_class(c, discovery_and_eval_lists["discovery_samples"][str(c)], discovery, model, device)
        torch.save(result, out_dir / f"class_{c:02d}.pt")
        results.append(result)

        cells = ""
        for t in C.CACHED_TARGETS:
            empty = int((result["region_pixels"][t] == 0).sum())
            cells += f" {np.nanmean(result['f_values'][t].numpy()):8.2f} ({empty:2d})"
        print(f"{c:3d} {VOC_CLASSES[c]:12s} {len(result['ids']):5d}{cells} {time.time() - start:5.0f}s", flush=True)
        for t in C.CACHED_TARGETS:
            for image_id, pixels in zip(result["ids"], result["region_pixels"][t]):
                if pixels == 0:
                    print(f"      NOTE: {image_id}: empty '{t}' region (kept, zero {t} scores)")

    for target in C.CACHED_TARGETS:
        layer_summary(results, target)
    utils.write_run_info(out_dir)
    print(f"\nSaved {len(results)} class files to {out_dir}")


if __name__ == "__main__":
    main()
