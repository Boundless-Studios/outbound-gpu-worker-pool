"""Explicit machine recipes; never inferred from VRAM alone.

The bounded 16GB graph and the 12GB subject-image graph are opt-in, outside the
standard template directory.
A profile's fingerprint binds discovery, lease eligibility and execution.
Optional features are separate runtime versions, outside that fingerprint, so
declaring one never makes an already-registered worker ineligible for the
jobs it already serves; only a job that requires the feature asks for it.

A profile is either a ComfyUI template recipe (`kind='comfy'`) or an engine
recipe (`kind='engine'`, see `engine_recipes`). A profile that is not
`qualified`, or an engine profile with an unset pin, is refused at enrolment,
at worker registration and at dispatch.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .comfy import PACKAGED_TEMPLATES_DIRECTORY, TemplateRegistry, ComfyWorkflowPlugin
from .contracts import ProfileNotQualified
from .engine_recipes import (
    MESH3D, MESH3D_CLEAN_CAPABILITY, MESH3D_MESH_CAPABILITY, MESH3D_RECIPE, EngineRecipe,
    engine_recipe_version,
)

LIGHTX8 = 'h3-16gb-lightx8'
STANDARD = 'h3-24gb-standard'
# Subject images only: the standard Klein v2 graph, fixed at 1024², 20 steps and
# at most three references, under its own capability so a 12GB card never leases
# a 16GB subject job.
SUBJECT12 = 'flux2-klein-subject-12gb'
RECIPE_KEY = 'magic_recipe'
CONFIG_KEY = 'magic_config'
PROFILE_DIRECTORY = PACKAGED_TEMPLATES_DIRECTORY / 'machine_profiles'
LIGHTX8_CAPABILITY = 'video.minimax_h3.text_to_video.w4a8_16gb.lightx8.v1'
STANDARD_CAPABILITY = 'video.minimax_h3.text_to_video.v1'
SUBJECT12_CAPABILITY = 'image.flux2_klein.subject.12gb.v1'
PROFILE_CAPABILITIES = {LIGHTX8: LIGHTX8_CAPABILITY, STANDARD: STANDARD_CAPABILITY,
                        SUBJECT12: SUBJECT12_CAPABILITY}
# Flags that keep every model resident; the 12GB subject measurement ran without them.
SUBJECT12_REFUSED_FLAGS = ('--gpu-only', '--highvram')
# Closed-loop (first frame == last frame) conditioning, required only by such jobs.
LAST_FRAME_FEATURE_KEY = 'magic_feature_last_frame'


# The inputs a recipe-bound worker attests and the coordinator compares at
# completion. Every profile so far attests the same six.
H3_ATTESTED_INPUTS = ('steps', 'width', 'height', 'length', 'fps', 'seed')
# A runtime may serve several profiles. The first keeps the plain recipe and
# configuration keys, so jobs pinned before multi-profile workers still match;
# each further profile registers them under `<key>@<profile>`.
NAMESPACE_SEPARATOR = '@'


@dataclass(frozen=True)
class MachineProfile:
    """One explicit recipe: what it runs, how it is checked and what it attests.

    A new profile is one entry in PROFILES. Its description is fingerprinted
    into the recipe version, so changing any described value makes every
    registered worker of that profile ineligible for the jobs it serves.

    An engine profile has no template: `engine` describes it, `describe` is
    None and `capability_id` is its first capability (see `capability_ids`).
    """
    name: str
    capability_id: str
    # 'standard' templates ship in the packaged directory; 'profile' ones are
    # opt-in, under machine_profiles/. Empty for an engine profile.
    template_directory: str
    describe: Callable[[Any, list[str]], dict[str, Any]] | None
    attested_inputs: tuple[str, ...] = H3_ATTESTED_INPUTS
    # May share a runtime with other templates and profiles (mixed mode).
    mixed_mode: bool = True
    # Raises ValueError when the live ComfyUI launch flags break the recipe.
    check_launch: Callable[[list[str], dict[str, Any]], None] | None = None
    # Optional features (outside the fingerprint) from live object_info.
    features: Callable[[dict[str, Any]], dict[str, str]] | None = None
    # 'comfy' (a ComfyUI template) or 'engine' (a pinned engine release).
    kind: str = 'comfy'
    engine: EngineRecipe | None = None
    # False keeps the profile inert: no enrolment, registration or dispatch.
    qualified: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ('comfy', 'engine'):
            raise ValueError(f'{self.name}: unknown profile kind {self.kind!r}')
        if (self.kind == 'engine') != (self.engine is not None) or (self.describe is None) != (self.kind == 'engine'):
            raise ValueError(f'{self.name}: an engine profile has an engine recipe and no template')
        if self.qualified and self.engine is not None and self.engine.missing_pins():
            raise ValueError(f'{self.name} cannot be qualified with unset pins: '
                             f'{", ".join(self.engine.missing_pins())}')

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return self.engine.capability_ids if self.engine is not None else (self.capability_id,)


def unusable_reason(profile: str) -> str | None:
    """Why a profile may not enrol, register or run; None when it may."""
    spec = _profile(profile)
    missing = spec.engine.missing_pins() if spec.engine is not None else ()
    if spec.qualified and not missing:
        return None
    detail = f'unset pins: {", ".join(missing)}' if missing else 'qualified is False'
    return (f'{profile} is not qualified yet ({detail}); it cannot be enrolled, registered '
            f'or dispatched until its qualification sets the pins and marks it qualified')


def require_qualified(profile: str) -> None:
    reason = unusable_reason(profile)
    if reason is not None:
        raise ProfileNotQualified(reason)


def unqualified() -> dict[str, Any]:
    """Every unusable profile's name, capabilities and recipe version.

    The coordinator refuses registrations, submissions and leases that touch
    any of them. Only unusable profiles are described, so this stays cheap.
    """
    names = [name for name in PROFILES if unusable_reason(name) is not None]
    return {
        'profiles': frozenset(names),
        'capability_ids': frozenset(c for name in names for c in PROFILES[name].capability_ids),
        'recipe_versions': frozenset(describe_profile(name)['recipe_version'] for name in names),
    }


def registry_for(profile: str) -> TemplateRegistry:
    spec = _profile(profile)
    if spec.kind != 'comfy':
        raise ValueError(f'{profile} is an engine profile; it has no ComfyUI template')
    directory = PROFILE_DIRECTORY if spec.template_directory == 'profile' else PACKAGED_TEMPLATES_DIRECTORY
    return TemplateRegistry([TemplateRegistry.from_directory(directory).template(spec.capability_id)])


def _profile(profile: str) -> MachineProfile:
    try:
        return PROFILES[profile]
    except KeyError:
        raise ValueError(f'Unknown machine profile: {profile}') from None


def describe_profile(profile: str) -> dict[str, Any]:
    spec = _profile(profile)
    if spec.engine is not None:
        # Qualification is part of the fingerprint, so a job pinned to the
        # provisional recipe can never match a qualified worker.
        description = {**spec.engine.description(profile), 'qualified': spec.qualified}
        return {**description, 'recipe_version': engine_recipe_version(description)}
    template = registry_for(profile).template(spec.capability_id)
    graph = template.graph
    models = sorted({str(v) for node in graph.values() for k, v in node['inputs'].items()
                     if k in {'unet_name', 'clip_name', 'vae_name', 'lora_name'}})
    description = spec.describe(template, models)
    fingerprint = hashlib.sha256(json.dumps({'description': description, 'graph': graph},
                                           sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {**description, 'recipe_version': fingerprint}


def _subject12_description(template, models: list[str]) -> dict[str, Any]:
    return {
        'profile': SUBJECT12, 'capability_id': SUBJECT12_CAPABILITY,
        'template_version': template.template_version,
        'models': models, 'steps': 20, 'min_vram_mb': 12000,
        'sampler': 'euler', 'scheduler': 'Flux2Scheduler',
        'diffusion_precision': 'FP8 (stored)', 'text_encoder_precision': 'BF16',
        'bounds': {'width': 1024, 'height': 1024, 'references': len(template.image_slots)},
        'refused_flags': list(SUBJECT12_REFUSED_FLAGS),
        'evidence': 'RTX 4090 with VRAM held to 11,914 MiB free (and 10,900 MiB): 0, 1 and 3 '
                    'references at 1024x1024, 20 steps, byte-identical to the v2 graph, peak '
                    '10,218 MiB, worst 23.4 s; 2 references and a physical 12GB card unmeasured',
    }


def _video_description(profile: str, capability: str, template, models: list[str]) -> dict[str, Any]:
    # These fields and values are fingerprinted: changing any of them makes every
    # registered worker of the profile ineligible for the jobs it serves.
    return {
        'profile': profile, 'capability_id': capability,
        'template_version': template.template_version,
        'models': models, 'steps': 8 if profile == LIGHTX8 else 20,
        'min_vram_mb': 16000 if profile == LIGHTX8 else 24000,
        'sampler': 'res_multistep', 'scheduler': 'simple',
        'diffusion_precision': 'W4A8 mixed' if profile == LIGHTX8 else 'INT8 ConvRot',
        'text_encoder_precision': 'NVFP4/AWQ',
        'video_vae_precision': 'INT8 ConvRot' if profile == LIGHTX8 else 'FP16',
        'adapter': 'LightX2V FL2V 8-step v1.0 768p' if profile == LIGHTX8 else None,
        'adapter_strength': 1.0 if profile == LIGHTX8 else None,
        'reserve_vram_gb': 0 if profile == LIGHTX8 else None,
        'bounds': {'width': 512, 'height': 512, 'duration_seconds': 4, 'fps': 24}
                  if profile == LIGHTX8 else {'width': 1344, 'height': 1344, 'duration_seconds': 15},
        'h3_attention': 'comfy kitchen attention' if profile == LIGHTX8 else 'worker-configured',
        'process_attention': 'sage' if profile == LIGHTX8 else 'worker-configured',
        'audio_decode': profile != LIGHTX8,
        'evidence': 'RTX 4080 walk comparison; user selected B' if profile == LIGHTX8
                    else 'Existing standard template; no 24GB benchmark established',
    }


def profile_features(profile: str, object_info: dict[str, Any]) -> dict[str, str]:
    """Optional features this worker can actually run, as runtime versions."""
    spec = _profile(profile)
    return spec.features(object_info) if spec.features else {}


def _standard_features(object_info: dict[str, Any]) -> dict[str, str]:
    """Last-frame conditioning, declared only while it can actually run.

    Only the standard route declares it, and only while both its template wires
    a `last_frame` image slot into the H3 node's `last_frame` input and the live
    ComfyUI node (`object_info`) accepts that input. An older node that lacks it
    would lease a closed-loop job and fail inside ComfyUI.
    """
    template = registry_for(STANDARD).template(STANDARD_CAPABILITY)
    slot = next((s for s in template.image_slots if s.name == 'last_frame'), None)
    wired = slot is not None and any(
        node['class_type'] == 'MiniMaxH3ImageToVideo'
        and node['inputs'].get('last_frame') == [slot.node_id, 0]
        for node in template.graph.values())
    live = object_info.get('MiniMaxH3ImageToVideo', {}).get('input', {})
    accepts = 'last_frame' in {**live.get('required', {}), **live.get('optional', {})}
    return {LAST_FRAME_FEATURE_KEY: '1'} if wired and accepts else {}


def _check_subject12_launch(argv: list[str], system: dict[str, Any]) -> None:
    if any(flag in argv for flag in SUBJECT12_REFUSED_FLAGS):
        raise ValueError('12GB subject recipe refuses --gpu-only and --highvram; '
                         'it was measured with models offloaded between stages')


def _check_lightx8_launch(argv: list[str], system: dict[str, Any]) -> None:
    required = ['--use-sage-attention', '--enable-dynamic-vram', '--fast-disk', '--disable-pinned-memory']
    if any(flag not in argv for flag in required) or '--reserve-vram' not in argv:
        raise ValueError('16GB recipe requires Sage, dynamic VRAM, fast disk, disabled pinned memory and reserve-vram 0')
    if argv[argv.index('--reserve-vram')+1] not in ('0', '0.0'):
        raise ValueError('16GB recipe requires reserve-vram 0')
    if '+cu13' not in system.get('pytorch_version', ''):
        raise ValueError('16GB recipe requires a CUDA 13 PyTorch build')


async def inspect_runtime(profile: str, client: Any, base_url: str) -> dict[str, str]:
    """Fail before advertising capabilities when models/nodes/runtime are missing."""
    stats_response = await client.get(base_url.rstrip('/') + '/system_stats')
    stats_response.raise_for_status()
    stats = stats_response.json()
    response = await client.get(base_url.rstrip('/') + '/object_info')
    response.raise_for_status()
    info = response.json()
    description = describe_profile(profile)
    registry = registry_for(profile)
    template = registry.template(description['capability_id'])
    for node in template.graph.values():
        node_info = info.get(node['class_type'])
        if node_info is None:
            raise ValueError(f"Required ComfyUI node missing: {node['class_type']}")
        inputs = {**node_info.get('input', {}).get('required', {}),
                  **node_info.get('input', {}).get('optional', {})}
        for key, value in node['inputs'].items():
            if key not in {'unet_name', 'clip_name', 'vae_name', 'lora_name'}:
                continue
            choices = inputs.get(key, [[]])[0]
            if value not in choices:
                raise ValueError(f'Required model missing: {value}')
    devices = [d for d in stats.get('devices', []) if d.get('type') == 'cuda']
    if not devices or max(d.get('vram_total', 0) for d in devices)//(1024*1024) < description['min_vram_mb']:
        raise ValueError('No CUDA GPU with enough VRAM for the configured profile')
    system = stats['system']
    argv = system.get('argv', [])
    check_launch = _profile(profile).check_launch
    if check_launch:
        check_launch(argv, system)
    versions = {RECIPE_KEY: description['recipe_version'],
                'comfyui': str(system.get('comfyui_version', 'unknown')),
                'pytorch': str(system.get('pytorch_version', 'unknown')),
                'magic_profile': profile,
                'process_attention': ('sage' if '--use-sage-attention' in argv else
                                      'kitchen' if '--use-ck-attention' in argv else 'comfy-default'),
                'dynamic_vram': str('--enable-dynamic-vram' in argv).lower(),
                'pinned_memory_disabled': str('--disable-pinned-memory' in argv).lower(),
                'reserve_vram': str(argv[argv.index('--reserve-vram')+1]) if '--reserve-vram' in argv else 'comfy-default'}
    config = {'versions': versions, 'argv': argv, 'models': description['models']}
    versions[CONFIG_KEY] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    # After the configuration fingerprint, so declaring a feature changes neither it
    # nor the recipe version.
    versions.update(profile_features(profile, info))
    return versions


def recipe_input_defaults(recipe_version: str) -> dict[str, Any]:
    """The template defaults a profile's worker fills in for omitted inputs.

    A recipe-bound worker attests its effective inputs, defaults included, so
    the coordinator compares an omitted field with the recipe's default rather
    than with nothing. Unknown recipes have none.
    """
    return dict(_recipe_input_defaults(recipe_version))


@lru_cache(maxsize=8)
def _recipe_input_defaults(recipe_version: str) -> tuple[tuple[str, Any], ...]:
    profile = _profile_for_recipe(recipe_version)
    # An engine recipe has no defaults: its plugin requires every attested setting.
    if profile is None or PROFILES[profile].kind != 'comfy':
        return ()
    template = registry_for(profile).template(PROFILES[profile].capability_id)
    return tuple((entry.name, entry.default) for entry in template.inputs if entry.default is not None)


@lru_cache(maxsize=8)
def _profile_for_recipe(recipe_version: str) -> str | None:
    for profile in PROFILES:
        if describe_profile(profile)['recipe_version'] == recipe_version:
            return profile
    return None


def recipe_attested_inputs(recipe_version: str, capability_id: str | None = None) -> tuple[str, ...]:
    """The effective inputs the coordinator compares for a recipe-bound job.

    Unknown recipes keep the original six, so a coordinator never compares
    less than it did before profiles declared their own. An engine recipe
    attests per capability; for a capability it does not serve, every input
    any of its capabilities attests.
    """
    profile = _profile_for_recipe(recipe_version)
    if profile is None:
        return H3_ATTESTED_INPUTS
    spec = PROFILES[profile]
    if spec.engine is not None:
        stage = spec.engine.stage_for(capability_id) if capability_id else None
        if stage is not None:
            return stage.attested_inputs
    return spec.attested_inputs


def recipe_key(profile: str | None = None, key: str = RECIPE_KEY) -> str:
    """The runtime-versions key a profile's recipe (or configuration) is under.

    None means the runtime's first profile, which keeps the plain key.
    """
    return key if profile is None else f'{key}{NAMESPACE_SEPARATOR}{profile}'


def pinned_recipe(runtime_versions: dict[str, str]) -> str | None:
    """The recipe a job's requirements pin, under the plain or a namespaced key."""
    if RECIPE_KEY in runtime_versions:
        return runtime_versions[RECIPE_KEY]
    prefix = RECIPE_KEY + NAMESPACE_SEPARATOR
    return next((v for k, v in sorted(runtime_versions.items()) if k.startswith(prefix)), None)


def namespaced_versions(profile: str, versions: dict[str, str]) -> dict[str, str]:
    """A further profile's own keys, for registration beside the first profile's.

    Shared runtime facts (ComfyUI and PyTorch versions, launch flags) are the
    same for every profile of one runtime and stay under the first profile's
    plain keys; only the recipe, configuration and features are per profile.
    """
    own = {RECIPE_KEY: versions[RECIPE_KEY], CONFIG_KEY: versions[CONFIG_KEY],
           'magic_profile': versions['magic_profile'],
           **{k: v for k, v in versions.items() if k.startswith('magic_feature_')}}
    return {recipe_key(profile, key): value for key, value in own.items()}


def attested_versions(profile: str, versions: dict[str, str]) -> dict[str, str]:
    """What a further profile's plugin reports at completion.

    The shared runtime facts (ComfyUI and PyTorch versions, launch flags) under
    their plain keys, as the worker registered them, so a job that also pins one
    of them still verifies; the profile's own recipe, configuration, profile and
    features under its namespaced keys. The plain recipe keys belong to the
    runtime's first profile and are never reported for another.
    """
    shared = {k: v for k, v in versions.items()
              if k not in (RECIPE_KEY, CONFIG_KEY, 'magic_profile') and not k.startswith('magic_feature_')}
    return {**shared, **namespaced_versions(profile, versions)}


class ProfileComfyPlugin(ComfyWorkflowPlugin):
    """Recheck the runtime for each job; a restart/config change fails closed."""
    def __init__(self, profile: str, versions: dict[str, str], *, client: Any, base_url: str,
                 namespaced: bool = False, **runtime: Any):
        # `runtime` carries the start command and startup timeout, so a profile
        # job brings a stopped ComfyUI back exactly as a generic job would.
        require_qualified(profile)
        super().__init__(registry_for(profile), client=client, base_url=base_url, **runtime)
        self.profile = profile
        self.versions = versions
        # A further profile of a multi-profile runtime attests under its
        # namespaced keys, which are what its jobs pin.
        self.namespaced = namespaced
        self.profile_client = client
        self.profile_url = base_url

    async def execute(self, context, request):
        # Inspection talks to ComfyUI, so it must be running first.
        await self.ensure_running()
        actual = await inspect_runtime(self.profile, self.profile_client, self.profile_url)
        if actual != self.versions:
            raise ValueError('ComfyUI configuration changed; restart worker and resolve a new recipe')
        from dataclasses import replace
        output = await super().execute(context, request)
        attested = attested_versions(self.profile, actual) if self.namespaced else dict(actual)
        return replace(output, diagnostics={**output.diagnostics, 'runtime_versions': attested,
                                           'effective_inputs': dict(request.inputs)})


def parse_profiles(value: str) -> list[str]:
    """`MVS_WORKER_PROFILE`: one profile, or several separated by commas.

    Every name must be a known, qualified profile and appear once; the first
    keeps the plain recipe keys.
    """
    profiles = [name.strip() for name in value.split(',') if name.strip()]
    for profile in profiles:
        _profile(profile)
    for profile in profiles:
        require_qualified(profile)
    if len(set(profiles)) != len(profiles):
        raise ValueError(f'MVS_WORKER_PROFILE repeats a profile: {value}')
    if len(profiles) > 1:
        refused = [p for p in profiles if not PROFILES[p].mixed_mode]
        if refused:
            raise ValueError(f'{", ".join(refused)} cannot share a runtime with other profiles')
    return profiles


async def prepare_environment(environment: dict[str, str]) -> dict[str, str]:
    """Validate an explicit deployment profile and return enrollment-safe settings."""
    import httpx
    profiles = parse_profiles(environment.get('MVS_WORKER_PROFILE', ''))
    if not profiles:
        return {}
    if PROFILES[profiles[0]].kind == 'engine':
        # An engine profile never shares a runtime (mixed_mode is False), so
        # parse_profiles has already refused it beside any other profile.
        from .engine import prepare_engine_environment
        return await prepare_engine_environment(profiles[0], environment)
    base_url = environment.get('OGWP_COMFY_URL', 'http://127.0.0.1:8188')
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(base_url.rstrip('/') + '/system_stats')
            response.raise_for_status()
        except (httpx.HTTPError, OSError):
            command = environment.get('OGWP_COMFY_START_COMMAND')
            if not command:
                raise ValueError('ComfyUI is not running; start it or configure OGWP_COMFY_START_COMMAND') from None
            import asyncio
            import shlex
            process = await asyncio.create_subprocess_exec(*shlex.split(command),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            if await asyncio.wait_for(process.wait(), timeout=240) != 0:
                raise ValueError('ComfyUI start command failed')
            for attempt in range(120):
                try:
                    response = await client.get(base_url.rstrip('/') + '/system_stats')
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if attempt == 119:
                        raise ValueError('ComfyUI did not become ready') from None
                    await asyncio.sleep(2)
        per_profile = {profile: await inspect_runtime(profile, client, base_url) for profile in profiles}
        versions = dict(per_profile[profiles[0]])
        for profile in profiles[1:]:
            versions.update(namespaced_versions(profile, per_profile[profile]))
        response = await client.get(base_url.rstrip('/') + '/system_stats')
        response.raise_for_status()
        device = max((d for d in response.json()['devices'] if d.get('type') == 'cuda'), key=lambda d:d['vram_total'])
    settings = {'MVS_WORKER_PROFILE': ','.join(profiles), 'OGWP_COMFY_URL': base_url,
            'OGWP_WORKER_VRAM_MB': str(device['vram_total']//(1024*1024)),
            'OGWP_WORKER_GPU_MODEL': device['name'],
            'OGWP_WORKER_RUNTIME_VERSIONS': json.dumps(versions, separators=(',', ':')),
            'OGWP_WORKER_CONCURRENCY': '1'}
    if len(profiles) > 1:
        # Each profile plugin re-verifies its own, un-namespaced versions.
        settings['MVS_WORKER_PROFILE_VERSIONS'] = json.dumps(per_profile, separators=(',', ':'))
    if environment.get('OGWP_COMFY_START_COMMAND'):
        settings['OGWP_COMFY_START_COMMAND'] = environment['OGWP_COMFY_START_COMMAND']
    return settings


PROFILES: dict[str, MachineProfile] = {
    STANDARD: MachineProfile(
        STANDARD, STANDARD_CAPABILITY, 'standard',
        lambda template, models: _video_description(STANDARD, STANDARD_CAPABILITY, template, models),
        features=_standard_features),
    LIGHTX8: MachineProfile(
        LIGHTX8, LIGHTX8_CAPABILITY, 'profile',
        lambda template, models: _video_description(LIGHTX8, LIGHTX8_CAPABILITY, template, models),
        mixed_mode=False, check_launch=_check_lightx8_launch),
    SUBJECT12: MachineProfile(
        SUBJECT12, SUBJECT12_CAPABILITY, 'profile', _subject12_description,
        check_launch=_check_subject12_launch),
    # Provisional and unqualified: refused at enrolment, registration and
    # dispatch until qualification sets its pins (see engine_recipes).
    MESH3D: MachineProfile(
        MESH3D, MESH3D_MESH_CAPABILITY, '', None,
        attested_inputs=tuple(dict.fromkeys(
            name for stage in MESH3D_RECIPE.stages for name in stage.attested_inputs)),
        mixed_mode=False, kind='engine', engine=MESH3D_RECIPE, qualified=False),
}
# Kept for callers that map a ComfyUI profile to its capability directly.
assert PROFILE_CAPABILITIES == {name: spec.capability_id for name, spec in PROFILES.items()
                                if spec.kind == 'comfy'}


def main() -> int:
    import asyncio
    import os
    import sys
    try:
        settings = asyncio.run(prepare_environment(dict(os.environ)))
        print(json.dumps(settings))
        return 0
    except Exception as exc:
        print(f'Machine profile check failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
