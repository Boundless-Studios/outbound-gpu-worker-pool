"""Explicit machine recipes; never inferred from VRAM alone.

The bounded 16GB graph is opt-in, outside the standard template directory.
A profile's fingerprint binds discovery, lease eligibility and execution.
Optional features are separate runtime versions, outside that fingerprint, so
declaring one never makes an already-registered worker ineligible for the
jobs it already serves; only a job that requires the feature asks for it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .comfy import PACKAGED_TEMPLATES_DIRECTORY, TemplateRegistry, ComfyWorkflowPlugin

LIGHTX8 = 'h3-16gb-lightx8'
STANDARD = 'h3-24gb-standard'
RECIPE_KEY = 'magic_recipe'
CONFIG_KEY = 'magic_config'
PROFILE_DIRECTORY = PACKAGED_TEMPLATES_DIRECTORY / 'machine_profiles'
LIGHTX8_CAPABILITY = 'video.minimax_h3.text_to_video.w4a8_16gb.lightx8.v1'
STANDARD_CAPABILITY = 'video.minimax_h3.text_to_video.v1'
# Closed-loop (first frame == last frame) conditioning, required only by such jobs.
LAST_FRAME_FEATURE_KEY = 'magic_feature_last_frame'


def registry_for(profile: str) -> TemplateRegistry:
    if profile == LIGHTX8:
        return TemplateRegistry.from_directory(PROFILE_DIRECTORY)
    if profile == STANDARD:
        standard = TemplateRegistry.from_directory(PACKAGED_TEMPLATES_DIRECTORY)
        return TemplateRegistry([standard.template(STANDARD_CAPABILITY)])
    raise ValueError(f'Unknown machine profile: {profile}')


def describe_profile(profile: str) -> dict[str, Any]:
    registry = registry_for(profile)
    capability = LIGHTX8_CAPABILITY if profile == LIGHTX8 else STANDARD_CAPABILITY
    template = registry.template(capability)
    graph = template.graph
    models = sorted({str(v) for node in graph.values() for k, v in node['inputs'].items()
                     if k in {'unet_name', 'clip_name', 'vae_name', 'lora_name'}})
    description = {
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
    fingerprint = hashlib.sha256(json.dumps({'description': description, 'graph': graph},
                                           sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {**description, 'recipe_version': fingerprint}


def profile_features(profile: str) -> dict[str, str]:
    """Optional features the profile's verified graph supports, as runtime versions.

    Only the standard route declares last-frame conditioning, and only while its
    template actually wires a `last_frame` image slot into the H3 node's
    `last_frame` input. The bounded 16GB route declares nothing here.
    """
    if profile != STANDARD:
        return {}
    template = registry_for(profile).template(STANDARD_CAPABILITY)
    slot = next((s for s in template.image_slots if s.name == 'last_frame'), None)
    wired = slot is not None and any(
        node['class_type'] == 'MiniMaxH3ImageToVideo'
        and node['inputs'].get('last_frame') == [slot.node_id, 0]
        for node in template.graph.values())
    return {LAST_FRAME_FEATURE_KEY: '1'} if wired else {}


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
    if profile == LIGHTX8:
        required = ['--use-sage-attention', '--enable-dynamic-vram', '--fast-disk', '--disable-pinned-memory']
        if any(flag not in argv for flag in required) or '--reserve-vram' not in argv:
            raise ValueError('16GB recipe requires Sage, dynamic VRAM, fast disk, disabled pinned memory and reserve-vram 0')
        if argv[argv.index('--reserve-vram')+1] not in ('0', '0.0'):
            raise ValueError('16GB recipe requires reserve-vram 0')
        if '+cu13' not in system.get('pytorch_version', ''):
            raise ValueError('16GB recipe requires a CUDA 13 PyTorch build')
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
    versions.update(profile_features(profile))
    return versions


class ProfileComfyPlugin(ComfyWorkflowPlugin):
    """Recheck the runtime for each job; a restart/config change fails closed."""
    def __init__(self, profile: str, versions: dict[str, str], *, client: Any, base_url: str):
        super().__init__(registry_for(profile), client=client, base_url=base_url)
        self.profile = profile
        self.versions = versions
        self.profile_client = client
        self.profile_url = base_url

    async def execute(self, context, request):
        actual = await inspect_runtime(self.profile, self.profile_client, self.profile_url)
        if actual != self.versions:
            raise ValueError('ComfyUI configuration changed; restart worker and resolve a new recipe')
        from dataclasses import replace
        output = await super().execute(context, request)
        return replace(output, diagnostics={**output.diagnostics, 'runtime_versions': dict(actual),
                                           'effective_inputs': dict(request.inputs)})


async def prepare_environment(environment: dict[str, str]) -> dict[str, str]:
    """Validate an explicit deployment profile and return enrollment-safe settings."""
    import httpx
    profile = environment.get('MVS_WORKER_PROFILE')
    if not profile:
        return {}
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
        versions = await inspect_runtime(profile, client, base_url)
        response = await client.get(base_url.rstrip('/') + '/system_stats')
        response.raise_for_status()
        device = max((d for d in response.json()['devices'] if d.get('type') == 'cuda'), key=lambda d:d['vram_total'])
    settings = {'MVS_WORKER_PROFILE': profile, 'OGWP_COMFY_URL': base_url,
            'OGWP_WORKER_VRAM_MB': str(device['vram_total']//(1024*1024)),
            'OGWP_WORKER_GPU_MODEL': device['name'],
            'OGWP_WORKER_RUNTIME_VERSIONS': json.dumps(versions, separators=(',', ':')),
            'OGWP_WORKER_CONCURRENCY': '1'}
    if environment.get('OGWP_COMFY_START_COMMAND'):
        settings['OGWP_COMFY_START_COMMAND'] = environment['OGWP_COMFY_START_COMMAND']
    return settings


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
