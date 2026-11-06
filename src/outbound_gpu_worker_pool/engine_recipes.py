"""Declared recipes for engine (non-ComfyUI) machine profiles.

An engine profile runs a pinned engine release (today AISmith-3D's
`aismith-engine`) instead of a ComfyUI template. It describes itself from the
recipe declared here: the engine release and commit, the digests of the
engine's sources and weights locks, the CUDA architecture, the Blender series,
the minimum VRAM, and per capability the engine stage, its inputs and
settings, the attested inputs and the deadline. That description is the
fingerprint (`recipe_version`).

This module is standard-library only, so a coordinator installed without the
agent extras can still refuse an unqualified engine profile.

`mesh3d-24gb-standard` is provisional. Its numbers come from AISmith-3D's own
notes and code, not from a Studio measurement, and its engine pins are not set.
`machine_profiles` marks it unqualified, and the pool refuses it at enrolment,
registration and dispatch until qualification sets the pins and the flag.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

MESH3D = 'mesh3d-24gb-standard'
MESH3D_MESH_CAPABILITY = 'mesh.trellis2.image_to_mesh.v1'
MESH3D_CLEAN_CAPABILITY = 'mesh.blender.clean.v1'
AISMITH_ENGINE = 'aismith-engine'
GLB_CONTENT_TYPE = 'model/gltf-binary'
# The engine's view roles (`backend/engine/stages.py` VIEW_ROLES).
VIEW_ROLES = ('front', 'back', 'left', 'right')
# `backend/app/quality_tiers.py` in AISmith-3D.
QUALITY_TIERS = ('draft', 'standard', 'high', 'ultra')
TARGET_TRIANGLES_RANGE = (1_000, 500_000)
# The engine runs a requested 768 at 1024 (`native.run_trellis_fp8`), so this
# profile accepts only the resolutions that run as requested.
MESH_RESOLUTIONS = (512, 1024)
MAX_SEED = 2**31 - 1


@dataclass(frozen=True)
class EngineStage:
    """One capability: the engine stage that serves it and what it attests."""

    capability_id: str
    stage: str
    inputs: tuple[str, ...]
    required_inputs: tuple[str, ...]
    # Every setting a job may carry: (name, ('choices', values)) or
    # (name, ('range', (minimum, maximum))). Integers only, never booleans,
    # unless the choices are strings.
    settings: tuple[tuple[str, tuple[str, tuple[Any, ...]]], ...]
    # Settings a job must carry: all of `required_settings`, and exactly one
    # of `one_of_settings` when that is not empty.
    required_settings: tuple[str, ...]
    one_of_settings: tuple[str, ...]
    attested_inputs: tuple[str, ...]
    deadline_seconds: int

    def description(self) -> dict[str, Any]:
        return {
            'capability_id': self.capability_id, 'stage': self.stage,
            'inputs': list(self.inputs), 'required_inputs': list(self.required_inputs),
            'settings': {name: {kind: list(values)} for name, (kind, values) in self.settings},
            'required_settings': list(self.required_settings),
            'one_of_settings': list(self.one_of_settings),
            'attested_inputs': list(self.attested_inputs),
            'deadline_seconds': self.deadline_seconds,
        }


@dataclass(frozen=True)
class EngineRecipe:
    """What an engine profile runs, as pins the worker must match exactly."""

    engine: str
    # None while provisional: a profile with any unset pin is unusable.
    engine_tag: str | None
    engine_commit: str | None
    sources_lock_sha256: str | None
    weights_lock_sha256: str | None
    cuda_architecture: str
    # A Blender series such as '4.2'; `Blender 4.2.<patch>` matches.
    blender_series: str
    min_vram_mb: int
    stages: tuple[EngineStage, ...]
    # Where each number came from, fingerprinted with the recipe.
    provenance: tuple[tuple[str, str], ...]

    PINS = ('engine_tag', 'engine_commit', 'sources_lock_sha256', 'weights_lock_sha256')

    def missing_pins(self) -> tuple[str, ...]:
        return tuple(pin for pin in self.PINS if not getattr(self, pin))

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return tuple(stage.capability_id for stage in self.stages)

    def stage_for(self, capability_id: str) -> EngineStage | None:
        return next((s for s in self.stages if s.capability_id == capability_id), None)

    def description(self, profile: str) -> dict[str, Any]:
        return {
            'profile': profile, 'kind': 'engine', 'engine': self.engine,
            'engine_tag': self.engine_tag, 'engine_commit': self.engine_commit,
            'sources_lock_sha256': self.sources_lock_sha256,
            'weights_lock_sha256': self.weights_lock_sha256,
            'cuda_architecture': self.cuda_architecture,
            'blender': f'{self.blender_series}.x LTS',
            'min_vram_mb': self.min_vram_mb,
            'capabilities': [stage.description() for stage in self.stages],
            'provenance': dict(self.provenance),
        }


def engine_recipe_version(description: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({'description': description}, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


MESH3D_RECIPE = EngineRecipe(
    engine=AISMITH_ENGINE,
    # Provisional: no AISmith-3D engine release is tagged yet. The engine-cli
    # branch head is not a stable pin, so these stay unset until qualification.
    engine_tag=None,
    engine_commit=None,
    sources_lock_sha256=None,
    weights_lock_sha256=None,
    cuda_architecture='8.9',
    blender_series='4.2',
    min_vram_mb=24_000,
    stages=(
        EngineStage(
            capability_id=MESH3D_MESH_CAPABILITY, stage='mesh',
            inputs=VIEW_ROLES, required_inputs=('front',),
            settings=(('resolution', ('choices', MESH_RESOLUTIONS)), ('seed', ('range', (0, MAX_SEED)))),
            required_settings=('resolution', 'seed'), one_of_settings=(),
            attested_inputs=('resolution', 'seed'),
            deadline_seconds=7_200,
        ),
        EngineStage(
            capability_id=MESH3D_CLEAN_CAPABILITY, stage='clean',
            inputs=('mesh', *VIEW_ROLES), required_inputs=('mesh',),
            settings=(('quality_tier', ('choices', QUALITY_TIERS)),
                      ('target_triangles', ('range', TARGET_TRIANGLES_RANGE))),
            required_settings=(), one_of_settings=('quality_tier', 'target_triangles'),
            attested_inputs=('quality_tier', 'target_triangles'),
            deadline_seconds=1_800,
        ),
    ),
    provenance=(
        ('status', 'provisional and unqualified; no number below was measured in Studio'),
        ('engine_pins', 'unset: AISmith-3D has no tagged engine release yet'),
        ('min_vram_mb', 'AISmith-3D notes, untested in Studio: docs/gpu-usage.md (TRELLIS.2 FP8 '
                        'stages CPU/GPU offload below 20 GB free; audited on an RTX 4090 24 GB) and '
                        'HANDOFF-UNIRIG.md (Flux.2 Klein view synthesis ~16.5 GB)'),
        ('cuda_architecture', 'AISmith-3D TRELLIS.2 native build contract (native-contract.json '
                              'cuda_architecture 8.9, RTX 4090); untested in Studio'),
        ('blender', 'AISmith-3D README: Blender 4.2 or later, reference install 4.2.23 LTS'),
        ('mesh_deadline_seconds', 'AISmith-3D backend/app/native.py run_trellis_fp8 timeout=7200; '
                                  'also the pool maximum'),
        ('clean_deadline_seconds', 'AISmith-3D backend/app/native.py has no whole-stage canonicalize '
                                   'timeout; its longest Blender steps (voxel remesh, bake) each cap '
                                   'at timeout=1800. Up to 3 remesh attempts plus bake, floor, '
                                   'projection and similarity could exceed it; unmeasured'),
        ('mesh_resolutions', 'AISmith-3D run_trellis_fp8 runs 768 at 1024, so only 512 and 1024 '
                             'are accepted'),
    ),
)

# Every engine recipe the package declares. A coordinator that cannot load
# `machine_profiles` refuses all of their capabilities.
ENGINE_RECIPES: dict[str, EngineRecipe] = {MESH3D: MESH3D_RECIPE}
