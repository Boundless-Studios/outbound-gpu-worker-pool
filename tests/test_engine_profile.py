"""The provisional 3D engine profile: its declared recipe and every refusal that keeps it inert."""
import json
import sys
from dataclasses import replace
from functools import partial

import httpx
import pytest

from outbound_gpu_worker_pool import (
    JobRequirements, JobStatus, ProfileNotQualified, WorkerCapability, WorkerRegistration, agent_main,
    machine_profiles,
)
from outbound_gpu_worker_pool.engine_recipes import (
    MESH3D, MESH3D_CLEAN_CAPABILITY, MESH3D_MESH_CAPABILITY, MESH3D_RECIPE, engine_recipe_version,
)
from outbound_gpu_worker_pool.machine_profiles import (
    CONFIG_KEY, PROFILES, RECIPE_KEY, STANDARD, MachineProfile, describe_profile, parse_profiles,
    prepare_environment, recipe_attested_inputs, recipe_input_defaults, recipe_key, registry_for,
)
from outbound_gpu_worker_pool.service import CompletionRejected
from test_agent_main import STATIC_ENVIRONMENT
from test_service import ECHO, OUTPUT_BYTES, WORKER_A, _harness, _manifest, _submission

NOT_QUALIFIED = f'{MESH3D} is not qualified yet'
TEST_PINS = {'engine_tag': 'engine-v0.0.0-test', 'engine_commit': 'a' * 40,
             'sources_lock_sha256': 'b' * 64, 'weights_lock_sha256': 'c' * 64}


def qualified_profile() -> MachineProfile:
    """The mesh3d profile as qualification would leave it: pins set, flag on. Tests only."""
    return replace(PROFILES[MESH3D], engine=replace(MESH3D_RECIPE, **TEST_PINS), qualified=True)


def test_the_engine_profile_describes_its_declared_recipe():
    description = describe_profile(MESH3D)
    assert description['kind'] == 'engine' and description['engine'] == 'aismith-engine'
    assert description['qualified'] is False
    assert all(description[pin] is None for pin in TEST_PINS)
    assert description['cuda_architecture'] == '8.9' and description['blender'] == '4.2.x LTS'
    assert description['min_vram_mb'] == 24000
    stages = {c['capability_id']: c for c in description['capabilities']}
    assert set(stages) == {MESH3D_MESH_CAPABILITY, MESH3D_CLEAN_CAPABILITY}
    assert stages[MESH3D_MESH_CAPABILITY]['attested_inputs'] == ['resolution', 'seed']
    assert stages[MESH3D_MESH_CAPABILITY]['deadline_seconds'] == 1800
    assert stages[MESH3D_CLEAN_CAPABILITY]['attested_inputs'] == ['quality_tier', 'target_triangles']
    assert stages[MESH3D_CLEAN_CAPABILITY]['deadline_seconds'] == 600
    # Every provisional number carries its source.
    assert {'min_vram_mb', 'cuda_architecture', 'blender', 'mesh_deadline_seconds',
            'clean_deadline_seconds', 'engine_pins'} <= set(description['provenance'])
    assert 'untested in Studio' in description['provenance']['min_vram_mb']
    recipe = description.pop('recipe_version')
    assert recipe == engine_recipe_version(description)


def test_the_fingerprint_covers_pins_and_qualification(monkeypatch):
    provisional = describe_profile(MESH3D)['recipe_version']
    monkeypatch.setitem(PROFILES, MESH3D, qualified_profile())
    qualified = describe_profile(MESH3D)['recipe_version']
    monkeypatch.setitem(PROFILES, MESH3D, replace(qualified_profile(), engine=replace(
        qualified_profile().engine, engine_commit='d' * 40)))
    assert len({provisional, qualified, describe_profile(MESH3D)['recipe_version']}) == 3


def test_an_engine_profile_cannot_be_qualified_with_unset_pins():
    with pytest.raises(ValueError, match='cannot be qualified with unset pins: engine_tag'):
        replace(PROFILES[MESH3D], qualified=True)


def test_an_engine_profile_has_no_template_and_no_defaults():
    with pytest.raises(ValueError, match='engine profile'):
        registry_for(MESH3D)
    assert recipe_input_defaults(describe_profile(MESH3D)['recipe_version']) == {}
    assert PROFILES[MESH3D].capability_ids == (MESH3D_MESH_CAPABILITY, MESH3D_CLEAN_CAPABILITY)
    assert MESH3D not in machine_profiles.PROFILE_CAPABILITIES


def test_attested_inputs_are_per_capability():
    recipe = describe_profile(MESH3D)['recipe_version']
    assert recipe_attested_inputs(recipe, MESH3D_MESH_CAPABILITY) == ('resolution', 'seed')
    assert recipe_attested_inputs(recipe, MESH3D_CLEAN_CAPABILITY) == ('quality_tier', 'target_triangles')
    # A capability the recipe does not serve compares everything it attests.
    assert recipe_attested_inputs(recipe, ECHO) == ('resolution', 'seed', 'quality_tier', 'target_triangles')


# Enrolment refusals.

@pytest.mark.parametrize('value', [MESH3D, f'{STANDARD},{MESH3D}', f'{MESH3D},{STANDARD}'])
def test_parse_profiles_refuses_the_unqualified_profile(value):
    with pytest.raises(ProfileNotQualified, match=NOT_QUALIFIED):
        parse_profiles(value)


@pytest.fixture
def no_runtime(monkeypatch):
    """Fail the test if enrolment reaches ComfyUI or the engine."""
    def unreachable(request):
        raise AssertionError(f'enrolment contacted {request.url}')

    monkeypatch.setattr(httpx, 'AsyncClient', partial(httpx.AsyncClient, transport=httpx.MockTransport(unreachable)))

    async def no_subprocess(*args, **kwargs):
        raise AssertionError('enrolment started a process')

    monkeypatch.setattr('asyncio.create_subprocess_exec', no_subprocess)


async def test_prepare_environment_refuses_before_touching_the_runtime(no_runtime):
    with pytest.raises(ProfileNotQualified, match=NOT_QUALIFIED):
        await prepare_environment({'MVS_WORKER_PROFILE': MESH3D})


async def test_prepare_mixed_profiles_refuses_before_touching_the_runtime(no_runtime):
    with pytest.raises(ProfileNotQualified, match=NOT_QUALIFIED):
        await agent_main.prepare_mixed_profiles(STATIC_ENVIRONMENT | {'MVS_WORKER_PROFILE': MESH3D})


def test_profile_plugins_refuse_the_unqualified_profile():
    from outbound_gpu_worker_pool.engine import AISmithEnginePlugin

    with pytest.raises(ProfileNotQualified, match=NOT_QUALIFIED):
        AISmithEnginePlugin(MESH3D, {RECIPE_KEY: describe_profile(MESH3D)['recipe_version']})
    with pytest.raises(ProfileNotQualified, match=NOT_QUALIFIED):
        machine_profiles.ProfileComfyPlugin(MESH3D, {}, client=None, base_url='http://127.0.0.1:8188')


def test_a_qualified_engine_profile_still_never_shares_a_runtime(monkeypatch):
    monkeypatch.setitem(PROFILES, MESH3D, qualified_profile())
    assert parse_profiles(MESH3D) == [MESH3D]
    with pytest.raises(ValueError, match='cannot share a runtime'):
        parse_profiles(f'{STANDARD},{MESH3D}')


# Coordinator refusals.

def _registration(*capabilities: str, **runtime_versions: str) -> WorkerRegistration:
    return WorkerRegistration(
        worker_id=WORKER_A.worker_id,
        capabilities=tuple(WorkerCapability(c, 'aismith-engine', '1') for c in capabilities),
        runtime_versions=dict(runtime_versions))


@pytest.mark.parametrize('registration', [
    _registration(MESH3D_MESH_CAPABILITY),
    _registration(ECHO, MESH3D_CLEAN_CAPABILITY),
    _registration(ECHO, magic_profile=MESH3D),
    _registration(ECHO, **{recipe_key(MESH3D): 'r' * 64}),
    _registration(ECHO, magic_recipe=describe_profile(MESH3D)['recipe_version']),
], ids=['mesh', 'clean-beside-echo', 'profile-name', 'namespaced-key', 'recipe'])
async def test_registration_advertising_the_unqualified_profile_is_refused(registration):
    harness = _harness()
    with pytest.raises(ProfileNotQualified, match='not qualified yet'):
        await harness.service.register_heartbeat(WORKER_A, registration)
    assert await harness.registry.get(WORKER_A.worker_id) is None
    # The same worker without it registers normally.
    await harness.service.register_heartbeat(WORKER_A, _registration(ECHO, python='3.13'))


def test_the_heartbeat_route_answers_422():
    from test_routes import _authorization, _harness as routes_harness

    harness = routes_harness()
    response = harness.client.post('/worker/v1/heartbeat', headers=_authorization('token-a'), json={
        'worker_id': 'worker-a', 'tenant_id': 'tenant-a',
        'capabilities': [{'capability_id': MESH3D_MESH_CAPABILITY, 'plugin_id': 'aismith-engine',
                          'plugin_version': '1'}]})
    assert response.status_code == 422
    assert response.json()['detail'] == {'reason': 'profile_not_qualified'}


@pytest.mark.parametrize('overrides', [
    {'capability_id': MESH3D_MESH_CAPABILITY},
    {'capability_id': MESH3D_CLEAN_CAPABILITY},
    {'requirements': JobRequirements(runtime_versions={RECIPE_KEY: describe_profile(MESH3D)['recipe_version']})},
    {'requirements': JobRequirements(runtime_versions={recipe_key(MESH3D): 'r' * 64})},
    {'requirements': JobRequirements(runtime_versions={'magic_profile': MESH3D})},
], ids=['mesh', 'clean', 'recipe', 'namespaced-key', 'profile-name'])
async def test_submit_refuses_jobs_for_the_unqualified_profile(overrides):
    harness = _harness()
    with pytest.raises(ProfileNotQualified, match='not qualified yet'):
        await harness.service.submit(_submission(**overrides))
    assert harness.jobs.records == {}


async def _seeded_around_the_refusals(jobs, registry, *, versions=None, assets=None):
    """A mesh job and an echo job stored, and a worker registered, directly in the stores."""
    if assets is not None:
        await assets.write_once('inputs/pool/echo.bin', b'echo', 'application/octet-stream')
    mesh = (await jobs.submit(_submission('mesh', capability_id=MESH3D_MESH_CAPABILITY))).record
    echo = (await jobs.submit(_submission('echo'))).record
    await registry.upsert(_registration(MESH3D_MESH_CAPABILITY, ECHO, **(versions or {})),
                          identity_subject=WORKER_A.subject)
    return mesh, echo


async def test_lease_never_hands_out_an_unqualified_job_from_memory():
    harness = _harness()
    mesh, echo = await _seeded_around_the_refusals(harness.jobs, harness.registry, assets=harness.assets)
    grant = await harness.service.lease(WORKER_A, (MESH3D_MESH_CAPABILITY, ECHO))
    assert grant.job_id == echo.job_id
    assert await harness.service.lease(WORKER_A, (MESH3D_MESH_CAPABILITY,)) is None
    assert harness.jobs.records[mesh.job_id].status is JobStatus.QUEUED
    assert harness.jobs.records[mesh.job_id].attempts == 0


async def test_a_worker_naming_the_unqualified_profile_leases_nothing_from_memory():
    harness = _harness()
    _, echo = await _seeded_around_the_refusals(harness.jobs, harness.registry,
                                                versions={'magic_profile': MESH3D})
    assert await harness.service.lease(WORKER_A, (ECHO,)) is None
    assert harness.jobs.records[echo.job_id].status is JobStatus.QUEUED


async def test_completion_of_an_unqualified_job_is_rejected():
    harness = _harness()
    mesh, _ = await _seeded_around_the_refusals(harness.jobs, harness.registry)
    # Leased around the service, as only a broken store could.
    record = await harness.jobs.lease(worker_id=WORKER_A.worker_id, capability_ids=(MESH3D_MESH_CAPABILITY,),
                                      lease_seconds=60)
    assert record.job_id == mesh.job_id
    await harness.assets.write_once(record.output_key, OUTPUT_BYTES, 'text/plain')
    grant = type('Grant', (), {'output_key': record.output_key, 'idempotency_key': record.idempotency_key,
                               'request_digest': record.request_digest})
    with pytest.raises(CompletionRejected, match='profile_not_qualified'):
        await harness.service.complete(WORKER_A, mesh.job_id, record.claim_token, _manifest(grant, OUTPUT_BYTES))
    assert harness.jobs.records[mesh.job_id].status is JobStatus.FAILED


async def test_without_the_profile_module_every_engine_capability_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, 'outbound_gpu_worker_pool.machine_profiles', None)
    monkeypatch.setitem(PROFILES, MESH3D, qualified_profile())
    harness = _harness()
    with pytest.raises(ProfileNotQualified):
        await harness.service.submit(_submission(capability_id=MESH3D_CLEAN_CAPABILITY))
    with pytest.raises(ProfileNotQualified):
        await harness.service.register_heartbeat(WORKER_A, _registration(ECHO, magic_profile=MESH3D))


# Attestation, with the profile as qualification would leave it.

def _engine_job(capability: str, payload: dict, recipe: str):
    pinned = {RECIPE_KEY: recipe, CONFIG_KEY: 'c' * 64}
    return _submission('engine', capability_id=capability, input_keys=('inputs/pool/front.png',),
                       output_key='outputs/pool/engine.glb', payload=payload,
                       requirements=JobRequirements(runtime_versions=pinned)), pinned


@pytest.mark.parametrize('capability, payload, effective, accepted', [
    (MESH3D_MESH_CAPABILITY, {'resolution': 1024, 'seed': 7}, {'resolution': 1024, 'seed': 7}, True),
    (MESH3D_MESH_CAPABILITY, {'resolution': 1024, 'seed': 7}, {'resolution': 1024, 'seed': 8}, False),
    (MESH3D_MESH_CAPABILITY, {'resolution': 512, 'seed': 7}, {'resolution': 1024, 'seed': 7}, False),
    (MESH3D_MESH_CAPABILITY, {'resolution': 1024, 'seed': 7}, {'resolution': 1024}, False),
    (MESH3D_CLEAN_CAPABILITY, {'quality_tier': 'high'}, {'quality_tier': 'high'}, True),
    (MESH3D_CLEAN_CAPABILITY, {'quality_tier': 'high'}, {'quality_tier': 'standard'}, False),
    # A derived value the job did not set is never reported as attested.
    (MESH3D_CLEAN_CAPABILITY, {'quality_tier': 'high'}, {'quality_tier': 'high', 'target_triangles': 80000}, False),
    (MESH3D_CLEAN_CAPABILITY, {'target_triangles': 120000}, {'target_triangles': 120000}, True),
], ids=['mesh', 'seed', 'resolution', 'missing-seed', 'clean', 'tier', 'derived', 'custom'])
async def test_the_coordinator_attests_engine_inputs(monkeypatch, capability, payload, effective, accepted):
    monkeypatch.setitem(PROFILES, MESH3D, qualified_profile())
    recipe = describe_profile(MESH3D)['recipe_version']
    harness = _harness()
    submission, pinned = _engine_job(capability, {'inputs': {'front': 'inputs/pool/front.png'}, **payload}, recipe)
    await harness.assets.write_once('inputs/pool/front.png', b'png', 'image/png')
    await harness.service.submit(submission)
    await harness.service.register_heartbeat(WORKER_A, replace(_registration(capability), runtime_versions=pinned))
    grant = await harness.service.lease(WORKER_A, (capability,))
    assert grant is not None and grant.job_id == submission.job_id
    await harness.assets.write_once(grant.output_key, b'glTF' + bytes(60), 'model/gltf-binary')
    complete = harness.service.complete(WORKER_A, grant.job_id, grant.claim_token, _manifest(
        grant, b'glTF' + bytes(60), diagnostics={'runtime_versions': pinned, 'effective_inputs': effective}))
    if accepted:
        await complete
        assert harness.jobs.records[grant.job_id].status is JobStatus.COMPLETED
    else:
        with pytest.raises(CompletionRejected, match='effective_inputs_mismatch'):
            await complete


async def test_the_coordinator_rejects_another_engine_configuration(monkeypatch):
    monkeypatch.setitem(PROFILES, MESH3D, qualified_profile())
    recipe = describe_profile(MESH3D)['recipe_version']
    harness = _harness()
    submission, pinned = _engine_job(MESH3D_MESH_CAPABILITY, {'inputs': {'front': 'inputs/pool/front.png'},
                                                             'resolution': 1024, 'seed': 7}, recipe)
    await harness.assets.write_once('inputs/pool/front.png', b'png', 'image/png')
    await harness.service.submit(submission)
    await harness.service.register_heartbeat(WORKER_A, replace(_registration(MESH3D_MESH_CAPABILITY),
                                                               runtime_versions=pinned))
    grant = await harness.service.lease(WORKER_A, (MESH3D_MESH_CAPABILITY,))
    await harness.assets.write_once(grant.output_key, b'glTF' + bytes(60), 'model/gltf-binary')
    with pytest.raises(CompletionRejected, match='recipe_configuration_mismatch'):
        await harness.service.complete(WORKER_A, grant.job_id, grant.claim_token, _manifest(
            grant, b'glTF' + bytes(60), diagnostics={'runtime_versions': {**pinned, CONFIG_KEY: 'd' * 64},
                                                     'effective_inputs': {'resolution': 1024, 'seed': 7}}))


def test_the_engine_recipe_fits_the_heartbeat_schema(monkeypatch):
    """Recipe and lock digests stay within the route's runtime-version bounds."""
    from outbound_gpu_worker_pool.routes import WorkerHeartbeatRequest

    monkeypatch.setitem(PROFILES, MESH3D, qualified_profile())
    WorkerHeartbeatRequest.model_validate({
        'worker_id': 'ilya-desktop',
        'capabilities': [{'capability_id': c, 'plugin_id': 'aismith-engine', 'plugin_version': '1'}
                         for c in PROFILES[MESH3D].capability_ids],
        'runtime_versions': {RECIPE_KEY: describe_profile(MESH3D)['recipe_version'],
                             'aismith_sources_lock': 'b' * 64, 'blender': 'Blender 4.2.23 LTS'},
    })
    assert json.dumps(describe_profile(MESH3D))
