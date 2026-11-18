"""Stage 10: the B-09 follow-on arm (D-14, D-16) and its matched control.

Encoder: frozen ``BAAI/bge-base-en-v1.5``, CLS pooling, L2-normalised by the
model's own pipeline, 768-d, 512-token window; inputs longer than the window
are rejected instead of silently truncated. Embeddings are cached per text,
like ``texttabpfn.encoder`` does for MiniLM (which stays unchanged).

Adapter: ``θᵢ + Wᵢ·P·e(t)``. ``P`` (384 x 768, no bias) is shared and learned,
initialised from a fixed seeded draw (``nn.Linear``'s default init), so both arms
start from identical weights. ``Wᵢ`` (64 x 192 x 384) starts at zero, so at
initialisation the module computes exactly the native rows. The rows are
identical across columns, like iteration 1. Trainable parameters:
64·192·384 + 384·768 = 4,718,592 + 294,912 = 5,013,504.

Training is stage 3's (``texttabpfn.training``) in lockstep: the same episode
stream (``training.make_episode``: scenes, n, rows, order and the D-09 story mix,
so the story arm reads exactly stage 3's story-arm texts), the same budget,
optimizer and schedule, and dev-only checkpoint selection on the same dev scenes
and sizes.

- Story arm: schema + story text (``episode.texts["story"]``).
- Control arm: schema text (``omitted``, ``episode.texts["schema"]``).
- Validation: each arm reads its training-condition text (story arm: correct
  story). The native model on the same rows is recorded once as a reference.

Run: ``uv run python -m texttabpfn.followon`` (resumable from the last
checkpoint). Writes ``config.json``, ``progress.csv``, ``validation.csv`` and,
when done, ``summary.json`` to ``results/10-followon-training/``; adapter weights
go to ``results/10-followon-training/weights/`` (``*.pt``, gitignored).
Report: ``python -m texttabpfn.report followon-training [--validation]``.
"""

from __future__ import annotations

import functools
import itertools
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Normalize, Pooling
from tabpfn.architectures import tabpfn_v2_5
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from texttabpfn import training
from texttabpfn.adapter import TextConditionedThinkingRows, episode_logits
from texttabpfn.model import binary_logits, checkpoint_rows, default_device, load_tabpfn
from texttabpfn.scenes import Scene, dev_scene_specs
from texttabpfn.stories import VARIANTS, scene_text, story_sample_specs
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

ENCODER_NAME = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768
MAX_TOKENS = 512
PROJ_DIM = 384
PROJECTION_SEED = 0

RESULTS_DIR = Path("results/10-followon-training")
ARMS = ("story", "control")
# Stage 3's episode text each arm reads, and its validation text variant.
EPISODE_TEXT = {"story": "story", "control": "schema"}
VALIDATION_VARIANT = {"story": "story", "control": "omitted"}
# Scene RNG stream for the corrupted texts of the token-length sample (as `report stories`).
_STORY_CHECK_STREAM = 20
COSINE_SCENES = 40


# --- Encoder --------------------------------------------------------------------


@functools.cache
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(ENCODER_NAME)


def token_lengths(texts: Sequence[str]) -> list[int]:
    """bge token counts including [CLS] and [SEP], without truncation."""
    ids = tokenizer()(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
    return [len(x) for x in ids]


class BgeEncoder:
    """Frozen bge-base-en-v1.5 (CLS pooling, L2-normalised); cached per text on the CPU as float32."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self._model = SentenceTransformer(ENCODER_NAME, device=str(device))
        self._model.eval()
        self._model.requires_grad_(False)
        modules = list(self._model)
        if self._model.max_seq_length != MAX_TOKENS:
            raise ValueError(f"expected a {MAX_TOKENS}-token window")
        if not (
            len(modules) == 3
            and isinstance(modules[1], Pooling)
            and modules[1].get_config_dict()["pooling_mode"] == "cls"
            and isinstance(modules[2], Normalize)
        ):
            raise ValueError("expected CLS pooling followed by L2 normalisation")
        self._cache: dict[str, torch.Tensor] = {}

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Embeddings ``(len(texts), 768)`` for texts within the token window."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            too_long = [n for n in token_lengths(missing) if n > MAX_TOKENS]
            if too_long:
                raise ValueError(f"text of {max(too_long)} tokens exceeds {MAX_TOKENS}")
            with torch.no_grad():
                emb = self._model.encode(missing, convert_to_tensor=True, batch_size=32)
            for text, e in zip(missing, emb.float().cpu(), strict=True):
                self._cache[text] = e.clone()
        return torch.stack([self._cache[t] for t in texts])

    def encode_one(self, text: str) -> torch.Tensor:
        """Embedding ``(768,)`` of one text."""
        return self.encode([text])[0]


# --- Adapter --------------------------------------------------------------------


class FollowOnThinkingRows(TextConditionedThinkingRows):
    """Thinking rows ``θ + W·P·e(t)``, shared over all columns.

    ``set_text`` takes a bge embedding ``(768,)`` or ``(B, 768)``, so
    ``adapter.conditioned`` and ``adapter.episode_logits`` work unchanged.
    """

    def __init__(
        self, rows_TE: torch.Tensor, text_dim: int = EMBED_DIM, proj_dim: int = PROJ_DIM
    ) -> None:
        super().__init__(rows_TE, text_dim=proj_dim)  # weight_TED: (T, E, proj_dim), zero
        self.text_dim = text_dim
        projection = torch.empty(proj_dim, text_dim, dtype=rows_TE.dtype)
        nn.init.kaiming_uniform_(
            projection, a=5**0.5, generator=torch.Generator().manual_seed(PROJECTION_SEED)
        )  # nn.Linear's default init, from a fixed seed
        self.projection_PD = nn.Parameter(projection.to(rows_TE.device))

    @classmethod
    def from_model(cls, model: tabpfn_v2_5.TabPFNV2p5) -> FollowOnThinkingRows:
        """A zero-initialised (W = 0) follow-on adapter around the checkpoint's rows."""
        return cls(checkpoint_rows(model))

    def rows(self) -> torch.Tensor:
        """Conditioned rows ``(b, T, E)``; ``b`` is 1 or the batch size."""
        if self._text_bD is None:
            raise RuntimeError("no text embedding set; call set_text first")
        z_bP = self._text_bD @ self.projection_PD.T
        shift_bTE = torch.einsum("tep,bp->bte", self.weight_TED, z_bP)
        return self.theta_TE[None] + shift_bTE


def trainable_parameters(adapter: nn.Module) -> int:
    return sum(p.numel() for p in adapter.parameters() if p.requires_grad)


def load_adapter(model, arm: str, out_dir: Path = RESULTS_DIR) -> FollowOnThinkingRows:
    """The arm's selected follow-on adapter (by dev validation loss), on the model's device."""
    adapter = FollowOnThinkingRows.from_model(model)
    saved = torch.load(selected_weights_path(out_dir, arm), map_location="cpu")
    adapter.load_state_dict(saved["state_dict"])
    return adapter


# --- Episodes, dev set and diagnostics -----------------------------------------


def episode_texts(ep: training.Episode) -> dict[str, str]:
    """Each arm's encoder input for a stage 3 episode."""
    return {arm: ep.texts[EPISODE_TEXT[arm]] for arm in ARMS}


class FollowOnDevSet(training.DevSet):
    """Stage 3's dev episodes; each arm reads its training-condition text through bge."""

    def __init__(self, settings: Settings, encoder: BgeEncoder, device: torch.device) -> None:
        super().__init__(settings, encoder, device)  # stage 3's arms and texts, embedded with bge
        self.embeddings = {arm: self.embeddings[EPISODE_TEXT[arm]] for arm in ARMS}


def max_sample_tokens() -> dict[str, int]:
    """bge token lengths over ``report stories``' fixed 1250-scene sample (every variant)."""
    stories, others = [], []
    for _, family, seed in story_sample_specs():
        scene = Scene(family, seed)
        rng = np.random.default_rng([family, seed, _STORY_CHECK_STREAM])
        lengths = token_lengths([scene_text(scene, v, rng) for v in VARIANTS])
        stories.append(lengths[0])
        others.append(max(lengths[1:]))
    return {"scenes": len(stories), "max_story": max(stories), "max_any": max(stories + others)}


def story_cosines(encoder: BgeEncoder, num_scenes: int = COSINE_SCENES) -> dict[str, float]:
    """Pairwise cosine of schema+story embeddings on the first dev scenes: all pairs,
    pairs with the same number of features p, and the mean residual norm after
    removing the mean embedding (the stage 3 MiniLM diagnostic)."""
    scenes = [Scene(*spec) for spec in dev_scene_specs(num_scenes)]
    emb = encoder.encode([scene_text(s, "story") for s in scenes])
    emb = emb / emb.norm(dim=1, keepdim=True)
    cos = emb @ emb.T
    pairs = list(itertools.combinations(range(len(scenes)), 2))
    same_p = [(i, j) for i, j in pairs if scenes[i].num_features == scenes[j].num_features]
    resid = (emb - emb.mean(0)).norm(dim=1)
    return {
        "scenes": len(scenes),
        "cos_mean": round(float(np.mean([cos[i, j] for i, j in pairs])), 4),
        "cos_same_p": round(float(np.mean([cos[i, j] for i, j in same_p])), 4),
        "same_p_pairs": len(same_p),
        "residual_norm": round(float(resid.mean()), 4),
    }


# --- Training -------------------------------------------------------------------


def _save_weights(path: Path, adapter: FollowOnThinkingRows, arm: str, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
    torch.save({"arm": arm, "step": step, "state_dict": state}, path)


def run(settings: Settings, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / CONFIG_JSON
    state_path = out_dir / WEIGHTS_DIR / STATE_FILE
    device = default_device()
    model = load_tabpfn(device)
    encoder = BgeEncoder(device)
    adapters = {arm: FollowOnThinkingRows.from_model(model) for arm in ARMS}
    if config_path.exists():
        saved = json.loads(config_path.read_text())["settings"]
        if saved != training._settings_json(settings):
            raise SystemExit(f"{config_path} holds different settings; refusing to resume")
    else:
        tokens = max_sample_tokens()
        if tokens["max_any"] > MAX_TOKENS:
            raise SystemExit(f"a sample text has {tokens['max_any']} bge tokens > {MAX_TOKENS}")
        adapter = adapters[ARMS[0]]
        config = {
            "settings": training._settings_json(settings),
            "arms": list(ARMS),
            "episode_text": EPISODE_TEXT,
            "validation_variant": VALIDATION_VARIANT,
            "story_mix_per_20": {v: k for v, k in training._MIX},
            "encoder": {
                "name": ENCODER_NAME,
                "pooling": "cls",
                "normalized": True,
                "dim": EMBED_DIM,
                "max_tokens": MAX_TOKENS,
                "sample_tokens": tokens,
                "dev_story_cosines": story_cosines(encoder),
            },
            "projection": {"shape": list(adapter.projection_PD.shape), "seed": PROJECTION_SEED},
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

    dev = FollowOnDevSet(settings, encoder, device)
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
        first = (step - 1) * settings.episodes_per_step
        episodes = [training.make_episode(settings, first + j) for j in range(settings.episodes_per_step)]
        texts = [episode_texts(ep) for ep in episodes]
        embedded = encoder.encode([t[arm] for t in texts for arm in ARMS]).view(len(episodes), len(ARMS), -1)
        for ep, emb in zip(episodes, embedded, strict=True):
            tensors = tuple(torch.as_tensor(a, device=device) for a in ep.arrays)
            for k, arm in enumerate(ARMS):
                logits = episode_logits(model, adapters[arm], emb[k], tensors)
                loss = F.cross_entropy(logits.reshape(-1, 2), tensors[3].reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss, arm {arm}, episode {ep.index}")
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
