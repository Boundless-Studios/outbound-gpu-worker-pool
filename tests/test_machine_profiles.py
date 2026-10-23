import copy
import json

from dataclasses import replace

import httpx
import pytest
from outbound_gpu_worker_pool import machine_profiles
from outbound_gpu_worker_pool.comfy import TemplateRegistry
from outbound_gpu_worker_pool.machine_profiles import (
    LIGHTX8, STANDARD, RECIPE_KEY, CONFIG_KEY, LAST_FRAME_FEATURE_KEY, STANDARD_CAPABILITY,
    describe_profile, inspect_runtime, profile_features, registry_for,
)


def runtime():
    stats = {'system': {'comfyui_version':'0.41.0', 'pytorch_version':'2.14.0+cu130',
              'argv':['--use-sage-attention','--enable-dynamic-vram','--fast-disk',
                      '--disable-pinned-memory','--reserve-vram','0']},
             'devices':[{'type':'cuda','name':'RTX 4080','vram_total':16384*1024**2}]}
    info = {}
    for node in registry_for(LIGHTX8).templates[0].graph.values():
        fields = info.setdefault(node['class_type'], {'input':{'required':{}}})['input']['required']
        for key,value in node['inputs'].items():
            if key in {'unet_name','clip_name','vae_name','lora_name'}:
                fields.setdefault(key,[[]])[0].append(value)
    return stats,info


async def inspect(stats, info):
    def handle(request):
        return httpx.Response(200,json=stats if request.url.path=='/system_stats' else info)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        return await inspect_runtime(LIGHTX8,client,'http://localhost:8189')


async def test_valid_profile_reports_version_and_changes_configuration_fingerprint():
    stats,info=runtime()
    versions=await inspect(stats,info)
    assert versions[RECIPE_KEY]==describe_profile(LIGHTX8)['recipe_version']
    assert len(versions[CONFIG_KEY])==64
    stats['system']['pytorch_version']='2.14.1+cu130'
    changed=await inspect(stats,info)
    assert versions[RECIPE_KEY]==changed[RECIPE_KEY]
    assert versions[CONFIG_KEY]!=changed[CONFIG_KEY]


@pytest.mark.parametrize('failure',['model','node','flags','vram','cuda'])
async def test_missing_dependencies_refuse_advertising(failure):
    stats,info=runtime()
    if failure=='model':
        info['UNETLoader']['input']['required']['unet_name'][0].clear()
    elif failure=='node': info.pop('ModelAttentionBackend')
    elif failure=='flags':stats['system']['argv'].remove('--use-sage-attention')
    elif failure=='vram':stats['devices'][0]['vram_total']=12*1024**3
    else:stats['system']['pytorch_version']='2.14.0+cu126'
    with pytest.raises(ValueError):await inspect(stats,info)


def test_bounded_profile_does_not_change_the_standard_route():
    b=describe_profile(LIGHTX8)
    standard=describe_profile(STANDARD)
    assert b['steps']==8 and b['min_vram_mb']==16000
    assert standard['steps']==20 and standard['min_vram_mb']==24000
    assert registry_for(STANDARD).template(b['capability_id']) is None
    entries={e.name:e for e in registry_for(LIGHTX8).templates[0].inputs}
    assert entries['steps'].minimum==entries['steps'].maximum==8
    assert entries['width'].maximum==entries['height'].maximum==512
    assert entries['length'].maximum==96


# Recipe fingerprints on main before the last-frame feature was declared. A
# changed value here would make every registered worker of that profile
# ineligible for jobs it already serves.
STANDARD_RECIPE_BEFORE_FEATURES = '54c9d39861b76e30226db5cc03e98b5c579bcef1809f746775e9e29ec3fb7f4f'
LIGHTX8_RECIPE_BEFORE_FEATURES = '2c2e440f34ea8037ea52d36ebd1743540481a277f695ddb46f1ed6cc138e446d'


def test_declaring_features_keeps_every_recipe_version():
    assert describe_profile(STANDARD)['recipe_version'] == STANDARD_RECIPE_BEFORE_FEATURES
    assert describe_profile(LIGHTX8)['recipe_version'] == LIGHTX8_RECIPE_BEFORE_FEATURES


def test_only_the_standard_route_declares_last_frame():
    _, info = standard_runtime()
    assert profile_features(STANDARD, info) == {LAST_FRAME_FEATURE_KEY: '1'}
    assert profile_features(LIGHTX8, info) == {}


@pytest.mark.parametrize('schema', ['absent', 'node-missing'])
async def test_last_frame_is_not_advertised_when_the_live_node_lacks_it(schema):
    stats, info = standard_runtime()
    if schema == 'absent':
        info['MiniMaxH3ImageToVideo']['input']['optional'].pop('last_frame')
    else:
        info['MiniMaxH3ImageToVideo']['input'].pop('optional')
    assert profile_features(STANDARD, info) == {}
    versions = await inspect_profile(STANDARD, stats, info)
    assert LAST_FRAME_FEATURE_KEY not in versions
    assert versions[RECIPE_KEY] == STANDARD_RECIPE_BEFORE_FEATURES


def test_last_frame_is_declared_only_while_the_template_wires_it(monkeypatch):
    template = registry_for(STANDARD).template(STANDARD_CAPABILITY)
    graph = copy.deepcopy(template.graph)
    h3 = next(node for node in graph.values() if node['class_type'] == 'MiniMaxH3ImageToVideo')
    del h3['inputs']['last_frame']
    unwired = TemplateRegistry([replace(template, graph=graph)])
    _, info = standard_runtime()
    monkeypatch.setattr(machine_profiles, 'registry_for', lambda profile: unwired)
    assert profile_features(STANDARD, info) == {}


def standard_runtime():
    stats = {'system': {'comfyui_version': '0.41.0', 'pytorch_version': '2.14.0+cu130', 'argv': []},
             'devices': [{'type': 'cuda', 'name': 'RTX 4090', 'vram_total': 24564 * 1024**2}]}
    info = {}
    for node in registry_for(STANDARD).template(STANDARD_CAPABILITY).graph.values():
        fields = info.setdefault(node['class_type'], {'input': {'required': {}}})['input']['required']
        for key, value in node['inputs'].items():
            if key in {'unet_name', 'clip_name', 'vae_name', 'lora_name'}:
                fields.setdefault(key, [[]])[0].append(value)
    # The live H3 node accepts the optional last frame, as current ComfyUI builds do.
    info['MiniMaxH3ImageToVideo']['input']['optional'] = {'last_frame': ['IMAGE']}
    return stats, info


async def inspect_profile(profile, stats, info):
    def handle(request):
        return httpx.Response(200, json=stats if request.url.path == '/system_stats' else info)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        return await inspect_runtime(profile, client, 'http://localhost:8188')


async def test_standard_runtime_advertises_last_frame_without_changing_its_fingerprints(monkeypatch):
    stats, info = standard_runtime()
    versions = await inspect_profile(STANDARD, stats, info)
    assert versions[LAST_FRAME_FEATURE_KEY] == '1'
    assert versions[RECIPE_KEY] == STANDARD_RECIPE_BEFORE_FEATURES
    monkeypatch.setattr(machine_profiles, 'profile_features', lambda profile, info: {})
    before = await inspect_profile(STANDARD, stats, info)
    assert LAST_FRAME_FEATURE_KEY not in before
    assert versions == {**before, LAST_FRAME_FEATURE_KEY: '1'}


async def test_bounded_runtime_versions_are_unchanged():
    stats, info = runtime()
    versions = await inspect(stats, info)
    assert LAST_FRAME_FEATURE_KEY not in versions
    assert versions[RECIPE_KEY] == LIGHTX8_RECIPE_BEFORE_FEATURES
