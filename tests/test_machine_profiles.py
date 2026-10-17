import copy
import json

import httpx
import pytest
from outbound_gpu_worker_pool.machine_profiles import (
    LIGHTX8, STANDARD, RECIPE_KEY, CONFIG_KEY, describe_profile, inspect_runtime, registry_for,
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
