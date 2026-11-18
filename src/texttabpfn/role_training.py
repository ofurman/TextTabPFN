"""Stage 8: episodic training of the role-token adapter and its matched control.

Both arms are a ``RoleTokenThinkingRows`` adapter (W zero-initialised, the same
seeded projection P) around the frozen TabPFN-2.5 backbone and the frozen MiniLM
encoder (D-11). They are trained in lockstep exactly like stage 3
(``texttabpfn.training``): the same episode stream (``training.make_episode``:
scenes, n, rows, order), the same budget, optimizer and schedule, and dev-only
checkpoint selection on the same dev scenes and sizes.

- Story arm: slot j holds the token of feature j's role sentence. The D-09 mix
  on the roles uses stage 3's per-episode assignment (``training.story_variant``,
  its own RNG stream): ``omitted`` = every slot neutral, ``one_wrong_fact`` =
  exactly one feature's role changed (``stories.one_wrong_fact_roles``), drawn
  from a further separate stream, so the rows stay identical to stage 3's.
- Control arm: slot j holds the neutral sentence's token for every feature.
- Validation: each arm reads its training-condition tokens (story arm: correct
  roles). The native model on the same rows is recorded once as a reference.

Run: ``uv run python -m texttabpfn.role_training`` (resumable from the last
checkpoint). Writes ``config.json``, ``progress.csv``, ``validation.csv`` and,
when done, ``summary.json`` to ``results/08-role-training/``; adapter weights go
to ``results/08-role-training/weights/`` (``*.pt``, gitignored, never distributed).
Report: ``python -m texttabpfn.report role-training [--validation]``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from texttabpfn import role_tokens, training
from texttabpfn.adapter import episode_logits
from texttabpfn.encoder import TextEncoder
from texttabpfn.model import binary_logits, default_device, load_tabpfn
from texttabpfn.role_tokens import RoleTokenThinkingRows, slot_tokens, variant_sentences
from texttabpfn.scenes import Scene, dev_scene_specs
from texttabpfn.training import (
    CONFIG_JSON,
    NATIVE,
    PROGRESS_CSV,
    PROGRESS_FIELDS,
    STATE_FILE,
    SUMMARY_JSON,
    VALIDATION_CSV,
    VALIDATION_FIELDS,
    WEIGHTS_DIR,
    Settings,
    selected_weights_path,
    weights_path,
)

RESULTS_DIR = Path("results/08-role-training")
ARMS = ("story", "control")
# The role variant each arm reads for a validation scene (its training condition).
VALIDATION_VARIANT = {"story": "story", "control": "omitted"}
# Stream for the one_wrong_fact role draw; stage 3's are 30 and 40-42.
_ROLE_STREAM = 50


@dataclasses.dataclass
class RoleEpisode:
    episode: training.Episode  # rows, n, scene and the story-arm variant
    sentences: dict[str, list[str]]  # arm -> per-feature sentences in column order


def make_role_episode(settings: Settings, index: int) -> RoleEpisode:
    """Stage 3's episode ``index`` with each arm's per-feature sentences."""
    ep = training.make_episode(settings, index)
    scene = Scene(ep.family, ep.seed)
    rng = np.random.default_rng([_ROLE_STREAM, settings.seed, index])
    sentences = {
        "story": variant_sentences(scene, ep.variant, rng),
        "control": variant_sentences(scene, "omitted"),
    }
    return RoleEpisode(ep, sentences)


class RoleDevSet(training.DevSet):
    """Stage 3's dev episodes; each arm reads its training-condition slot tokens."""

    def __init__(self, settings: Settings, encoder: TextEncoder, device: torch.device) -> None:
        super().__init__(settings, encoder, device)
        self.embeddings = {
            arm: [
                slot_tokens(encoder, variant_sentences(Scene(*spec), VALIDATION_VARIANT[arm]))
                for spec in self.specs
            ]
            for arm in ARMS
        }


def load_adapter(model, arm: str, out_dir: Path = RESULTS_DIR) -> RoleTokenThinkingRows:
    """The arm's selected role-token adapter (by dev validation loss), on the model's device."""
    adapter = RoleTokenThinkingRows.from_model(model)
    saved = torch.load(selected_weights_path(out_dir, arm), map_location="cpu")
    adapter.load_state_dict(saved["state_dict"])
    return adapter


def trainable_parameters(adapter: RoleTokenThinkingRows) -> int:
    return sum(p.numel() for p in adapter.parameters() if p.requires_grad)


def _save_weights(path: Path, adapter: RoleTokenThinkingRows, arm: str, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
    torch.save({"arm": arm, "step": step, "state_dict": state}, path)


def run(settings: Settings, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / CONFIG_JSON
    state_path = out_dir / WEIGHTS_DIR / STATE_FILE
    device = default_device()
    model = load_tabpfn(device)
    encoder = TextEncoder("cpu")
    adapters = {arm: RoleTokenThinkingRows.from_model(model) for arm in ARMS}
    if config_path.exists():
        saved = json.loads(config_path.read_text())["settings"]
        if saved != training._settings_json(settings):
            raise SystemExit(f"{config_path} holds different settings; refusing to resume")
    else:
        adapter = adapters[ARMS[0]]
        config = {
            "settings": training._settings_json(settings),
            "arms": list(ARMS),
            "validation_variant": VALIDATION_VARIANT,
            "story_mix_per_20": {v: k for v, k in training._MIX},
            "tokens": {
                "max_slots": adapter.max_slots,
                "slot_dim": adapter.projection_PD.shape[0],
                "neutral_sentence": role_tokens.NEUTRAL_SENTENCE,
                "projection_seed": role_tokens.PROJECTION_SEED,
            },
            "trainable_parameters": trainable_parameters(adapter),
            "optimizer": "Adam(betas=(0.9, 0.999), eps=1e-8, weight_decay=0), one per arm",
            "dev_scenes": [list(s) for s in dev_scene_specs(settings.val_scenes)],
            "device": str(device),
            "torch": torch.__version__,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    opts = {arm: torch.optim.Adam(adapters[arm].parameters(), lr=settings.lr) for arm in ARMS}
    episodes_done = {arm: 0 for arm in ARMS}
    start_step, elapsed = 0, 0.0
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        for arm in ARMS:
            adapters[arm].load_state_dict(state["adapters"][arm])
            opts[arm].load_state_dict(state["optimizers"][arm])
        episodes_done = state["episodes"]
        start_step, elapsed = state["step"], state["elapsed_s"]
        print(f"resuming from step {start_step}", flush=True)
    training._keep_rows(out_dir / PROGRESS_CSV, PROGRESS_FIELDS, start_step)
    training._keep_rows(out_dir / VALIDATION_CSV, VALIDATION_FIELDS, start_step)

    dev = RoleDevSet(settings, encoder, device)
    t_dev = time.time()

    def checkpoint(step: int) -> None:
        rows = []
        if step == 0:
            with torch.no_grad():
                native = {n: [] for n in dev.sizes}
                for episodes in dev.episodes:
                    for n, ep in episodes.items():
                        logits = binary_logits(model, *ep[:3])
                        native[n].append(float(F.cross_entropy(logits.reshape(-1, 2), ep[3].reshape(-1))))
            rows += [[0, NATIVE, n, len(dev.specs), f"{np.mean(v):.6f}"] for n, v in native.items()]
        for arm in ARMS:
            losses = dev.logloss(model, adapters[arm], arm)
            rows += [[step, arm, n, len(dev.specs), f"{v:.6f}"] for n, v in losses.items()]
            _save_weights(weights_path(out_dir, arm, step), adapters[arm], arm, step)
        training._append(out_dir / VALIDATION_CSV, rows)
        torch.save(
            {
                "step": step,
                "elapsed_s": elapsed,
                "episodes": episodes_done,
                "adapters": {a: adapters[a].state_dict() for a in ARMS},
                "optimizers": {a: opts[a].state_dict() for a in ARMS},
            },
            state_path,
        )
        print(f"checkpoint step {step}: " + " ".join(f"{r[1]}@{r[2]}={r[4]}" for r in rows), flush=True)

    if start_step == 0:
        checkpoint(0)
        print(f"dev set + step-0 validation: {time.time() - t_dev:.0f}s", flush=True)

    ckpts = set(training.checkpoint_steps(settings))
    window = {arm: [] for arm in ARMS}
    for step in range(start_step + 1, settings.steps + 1):
        t0 = time.time()
        lr = training.learning_rate(settings, step)
        for arm in ARMS:
            for group in opts[arm].param_groups:
                group["lr"] = lr
            opts[arm].zero_grad(set_to_none=True)
        step_loss = {arm: 0.0 for arm in ARMS}
        for j in range(settings.episodes_per_step):
            rep = make_role_episode(settings, (step - 1) * settings.episodes_per_step + j)
            tensors = tuple(torch.as_tensor(a, device=device) for a in rep.episode.arrays)
            for arm in ARMS:
                tokens = slot_tokens(encoder, rep.sentences[arm])
                logits = episode_logits(model, adapters[arm], tokens, tensors)
                loss = F.cross_entropy(logits.reshape(-1, 2), tensors[3].reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss, arm {arm}, episode {rep.episode.index}")
                (loss / settings.episodes_per_step).backward()
                step_loss[arm] += loss.item() / settings.episodes_per_step
                episodes_done[arm] += 1
        for arm in ARMS:
            opts[arm].step()
            window[arm].append(step_loss[arm])
        elapsed += time.time() - t0
        if step % settings.log_every == 0 or step == settings.steps:
            training._append(
                out_dir / PROGRESS_CSV,
                [
                    [step, arm, episodes_done[arm], f"{lr:.6g}", f"{np.mean(window[arm]):.6f}", f"{elapsed:.1f}"]
                    for arm in ARMS
                ],
            )
            print(
                f"step {step} lr {lr:.2e} "
                + " ".join(f"{a}={np.mean(window[a]):.4f}" for a in ARMS)
                + f" elapsed {elapsed:.0f}s",
                flush=True,
            )
            window = {arm: [] for arm in ARMS}
        if step in ckpts:
            checkpoint(step)

    selected = training.select_checkpoints(settings, out_dir, ARMS)
    for arm, (step, _) in selected.items():
        saved = torch.load(weights_path(out_dir, arm, step), map_location="cpu")
        torch.save(saved, selected_weights_path(out_dir, arm))
    summary = {
        "steps": settings.steps,
        "episodes": episodes_done,
        "train_wall_s": round(elapsed, 1),
        "selected": {
            arm: {"step": step, "val_logloss_mean": round(v, 6)} for arm, (step, v) in selected.items()
        },
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done: {summary}", flush=True)


def main(argv: list[str] | None = None) -> None:
    run(*training.parse_settings(argv, __doc__.splitlines()[0], RESULTS_DIR))


if __name__ == "__main__":
    main()
