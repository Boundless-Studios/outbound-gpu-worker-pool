"""The profile registry: unchanged recipes, per-profile attestation, several profiles per runtime."""
import json
from dataclasses import replace
from functools import partial

import httpx
import pytest

from outbound_gpu_worker_pool import JobRequirements, agent_main, machine_profiles
from outbound_gpu_worker_pool.machine_profiles import (
    CONFIG_KEY, H3_ATTESTED_INPUTS, LIGHTX8, PROFILES, RECIPE_KEY, STANDARD, STANDARD_CAPABILITY,
    SUBJECT12, SUBJECT12_CAPABILITY, ProfileComfyPlugin, describe_profile, namespaced_versions,
    parse_profiles, pinned_recipe, prepare_environment, recipe_attested_inputs, recipe_input_defaults,
    recipe_key, registry_for,
)

# Recipe versions computed on main before the registry existed. Any drift would
# make every registered worker ineligible for the jobs it already serves.
RECIPES_BEFORE_REGISTRY = {
    STANDARD: '54c9d39861b76e30226db5cc03e98b5c579bcef1809f746775e9e29ec3fb7f4f',
    LIGHTX8: '2c2e440f34ea8037ea52d36ebd1743540481a277f695ddb46f1ed6cc138e446d',
    SUBJECT12: 'dfc533aa38439c3a99e9104ff2a71f1c17877d62e39885eaa528cf860ee32aee',
}


@pytest.mark.parametrize('profile', sorted(RECIPES_BEFORE_REGISTRY))
def test_registry_keeps_every_recipe_version(profile):
    assert describe_profile(profile)['recipe_version'] == RECIPES_BEFORE_REGISTRY[profile]
    assert registry_for(profile).template(PROFILES[profile].capability_id)


def test_unknown_profiles_are_refused():
    with pytest.raises(ValueError, match='Unknown machine profile'):
        registry_for('mesh3d-24gb-standard')


def test_attested_inputs_follow_the_recipe():
    for profile in PROFILES:
        assert recipe_attested_inputs(describe_profile(profile)['recipe_version']) == H3_ATTESTED_INPUTS
    # An unknown recipe never compares less than before.
    assert recipe_attested_inputs('f' * 64) == H3_ATTESTED_INPUTS


def test_pinned_recipe_reads_plain_or_namespaced_keys():
    assert pinned_recipe({RECIPE_KEY: 'a', recipe_key(SUBJECT12): 'b'}) == 'a'
    assert pinned_recipe({recipe_key(SUBJECT12): 'b', CONFIG_KEY: 'c'}) == 'b'
    assert pinned_recipe({'comfyui': '0.41'}) is None
    assert recipe_key(SUBJECT12) == f'magic_recipe@{SUBJECT12}'
    assert recipe_key(SUBJECT12, CONFIG_KEY) == f'magic_config@{SUBJECT12}'


@pytest.mark.parametrize('value, error', [
    ('nope', 'Unknown machine profile'),
    (f'{STANDARD},{STANDARD}', 'repeats'),
    (f'{STANDARD},{LIGHTX8}', 'cannot share a runtime'),
])
def test_parse_profiles_refuses_bad_lists(value, error):
    with pytest.raises(ValueError, match=error):
        parse_profiles(value)


def test_parse_profiles_keeps_order():
    assert parse_profiles(f' {STANDARD} , {SUBJECT12} ') == [STANDARD, SUBJECT12]
    assert parse_profiles('') == []


def _combined_runtime():
    """A 24GB ComfyUI that has every node and model of both the standard and 12GB recipes."""
    stats = {'system': {'comfyui_version': '0.41.0', 'pytorch_version': '2.14.0+cu130', 'argv': ['--listen']},
             'devices': [{'type': 'cuda', 'name': 'NVIDIA GeForce RTX 4090', 'vram_total': 24564 * 1024**2}]}
    info: dict = {}
    for profile, capability in ((STANDARD, STANDARD_CAPABILITY), (SUBJECT12, SUBJECT12_CAPABILITY)):
        for node in registry_for(profile).template(capability).graph.values():
            fields = info.setdefault(node['class_type'], {'input': {'required': {}}})['input']['required']
            for key, value in node['inputs'].items():
                if key in {'unet_name', 'clip_name', 'vae_name', 'lora_name'}:
                    fields.setdefault(key, [[]])[0].append(value)
    return stats, info


@pytest.fixture
def comfy(monkeypatch):
    stats, info = _combined_runtime()

    def handle(request):
        return httpx.Response(200, json=stats if request.url.path == '/system_stats' else info)

    monkeypatch.setattr(httpx, 'AsyncClient', partial(httpx.AsyncClient, transport=httpx.MockTransport(handle)))
    return stats, info


async def test_two_profiles_register_side_by_side(comfy):
    settings = await prepare_environment({'MVS_WORKER_PROFILE': f'{STANDARD},{SUBJECT12}'})
    versions = json.loads(settings['OGWP_WORKER_RUNTIME_VERSIONS'])
    per_profile = json.loads(settings['MVS_WORKER_PROFILE_VERSIONS'])
    # The first profile keeps the plain keys, exactly as a single-profile worker registers them.
    assert versions[RECIPE_KEY] == RECIPES_BEFORE_REGISTRY[STANDARD]
    assert versions['magic_profile'] == STANDARD
    assert versions[CONFIG_KEY] == per_profile[STANDARD][CONFIG_KEY]
    # The second is namespaced.
    assert versions[recipe_key(SUBJECT12)] == RECIPES_BEFORE_REGISTRY[SUBJECT12]
    assert versions[recipe_key(SUBJECT12, CONFIG_KEY)] == per_profile[SUBJECT12][CONFIG_KEY]
    assert versions[f'magic_profile@{SUBJECT12}'] == SUBJECT12
    assert settings['MVS_WORKER_PROFILE'] == f'{STANDARD},{SUBJECT12}'
    assert settings['OGWP_WORKER_CONCURRENCY'] == '1'


async def test_a_single_profile_registers_as_before(comfy):
    settings = await prepare_environment({'MVS_WORKER_PROFILE': STANDARD})
    versions = json.loads(settings['OGWP_WORKER_RUNTIME_VERSIONS'])
    assert 'MVS_WORKER_PROFILE_VERSIONS' not in settings
    assert not any('@' in key for key in versions)


async def test_one_failing_profile_refuses_the_whole_runtime(comfy):
    stats, _info = comfy
    stats['system']['argv'] = ['--highvram']
    with pytest.raises(ValueError, match='12GB subject recipe refuses'):
        await prepare_environment({'MVS_WORKER_PROFILE': f'{STANDARD},{SUBJECT12}'})


async def test_agent_binds_each_profile_to_its_own_plugin(comfy):
    from test_agent_main import STATIC_ENVIRONMENT

    prepared, plugins = await agent_main.prepare_mixed_profiles(
        STATIC_ENVIRONMENT | {'OGWP_WORKER_PLUGINS': 'comfy-workflow', 'MVS_WORKER_PROFILE': f'{STANDARD},{SUBJECT12}'})
    agent = agent_main.build_agent_from_env(prepared, profile_plugin=plugins)
    standard, subject = plugins
    assert isinstance(standard, ProfileComfyPlugin) and not standard.namespaced
    assert isinstance(subject, ProfileComfyPlugin) and subject.namespaced
    assert agent._plugins[STANDARD_CAPABILITY] is standard
    assert agent._plugins[SUBJECT12_CAPABILITY] is subject
    assert standard.versions[RECIPE_KEY] == RECIPES_BEFORE_REGISTRY[STANDARD]
    assert subject.versions[RECIPE_KEY] == RECIPES_BEFORE_REGISTRY[SUBJECT12]
    with pytest.raises(ValueError, match='use prepare_mixed_profiles'):
        await agent_main.prepare_mixed_profile(
            STATIC_ENVIRONMENT | {'MVS_WORKER_PROFILE': f'{STANDARD},{SUBJECT12}'})


def test_namespaced_versions_carry_only_per_profile_facts():
    versions = {RECIPE_KEY: 'r', CONFIG_KEY: 'c', 'magic_profile': SUBJECT12, 'comfyui': '0.41.0',
                'magic_feature_x': '1'}
    assert namespaced_versions(SUBJECT12, versions) == {
        f'magic_recipe@{SUBJECT12}': 'r', f'magic_config@{SUBJECT12}': 'c',
        f'magic_profile@{SUBJECT12}': SUBJECT12, f'magic_feature_x@{SUBJECT12}': '1'}


@pytest.mark.parametrize('change, accepted', [(None, True), ('steps', False), ('runtime', False)])
async def test_coordinator_attests_a_namespaced_recipe(change, accepted):
    from outbound_gpu_worker_pool.service import CompletionRejected, JobStatus
    from test_service import OUTPUT_BYTES, WORKER_A, _harness, _leased, _manifest

    recipe = RECIPES_BEFORE_REGISTRY[SUBJECT12]
    pinned = {recipe_key(SUBJECT12): recipe, recipe_key(SUBJECT12, CONFIG_KEY): 'c' * 64}
    harness = _harness()
    grant = await _leased(harness)
    record = harness.jobs.records[grant.job_id]
    harness.jobs.records[grant.job_id] = replace(
        record, payload={'prompt': 'an orc'}, requirements=JobRequirements(runtime_versions=pinned))
    effective = {**recipe_input_defaults(recipe), 'prompt': 'an orc'}
    reported = dict(pinned)
    if change == 'steps':
        effective['steps'] += 1
    elif change == 'runtime':
        reported[recipe_key(SUBJECT12, CONFIG_KEY)] = 'd' * 64
    await harness.assets.write_once(grant.output_key, OUTPUT_BYTES, 'text/plain')
    complete = harness.service.complete(WORKER_A, grant.job_id, grant.claim_token, _manifest(
        grant, OUTPUT_BYTES, diagnostics={'runtime_versions': reported, 'effective_inputs': effective}))
    if accepted:
        await complete
        assert harness.jobs.records[grant.job_id].status == JobStatus.COMPLETED
    else:
        with pytest.raises(CompletionRejected):
            await complete
