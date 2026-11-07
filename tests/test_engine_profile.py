"""The 3D engine profile: its qualified recipe, and every refusal that keeps an unqualified one inert.

The shipped `mesh3d-24gb-standard` is qualified. Refusals are proven against a
test-only unqualified copy (`unqualified_profile`), so no unqualified variant
ships.
"""
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
PINS = ('engine_tag', 'engine_commit', 'sources_lock_sha256', 'weights_lock_sha256')
REAL_PINS = {'engine_tag': 'engine-v0.2.0', 'engine_commit': '4162e2e9d49b0fb7c97caeca174ecb468e6b1260',
             'sources_lock_sha256': '34e7c16907f20268952173fb4f0c5b69dfd674e60a5017502ac2efe4fe19549b',
             'weights_lock_sha256': 'c03f5a068628ebbff378c6f833a4ef04bd6fa06b0db6380a7514d6f5ac5c9d33'}


def unqualified_profile() -> MachineProfile:
    """The mesh3d profile with its pins unset and the flag off. Tests only; never shipped."""
    return replace(PROFILES[MESH3D], engine=replace(MESH3D_RECIPE, **dict.fromkeys(PINS)),
                   qualified=False)


@pytest.fixture
def unqualified(monkeypatch):
    profile = unqualified_profile()
    monkeypatch.setitem(PROFILES, MESH3D, profile)
    return describe_profile(MESH3D)['recipe_version']


def test_the_engine_profile_describes_its_declared_recipe():
    description = describe_profile(MESH3D)
    assert description['kind'] == 'engine' and description['engine'] == 'aismith-engine'
    assert description['qualified'] is True
    assert {pin: description[pin] for pin in PINS} == REAL_PINS
    assert description['cuda_architecture'] == '8.9' and description['blender'] == '4.2.x LTS'
    assert description['min_vram_mb'] == 24000
    stages = {c['capability_id']: c for c in description['capabilities']}
    assert set(stages) == {MESH3D_MESH_CAPABILITY, MESH3D_CLEAN_CAPABILITY}
    assert stages[MESH3D_MESH_CAPABILITY]['attested_inputs'] == ['resolution', 'seed']
    assert stages[MESH3D_MESH_CAPABILITY]['deadline_seconds'] == 1800
    assert stages[MESH3D_MESH_CAPABILITY]['weight_ids'] == [
        'trellis2-4b-fp8', 'trellis-image-large-ss-decoder', 'dinov3-vitl16', 'birefnet-general',
        'birefnet-general-lite']
    assert stages[MESH3D_CLEAN_CAPABILITY]['attested_inputs'] == ['quality_tier', 'target_triangles']
    assert stages[MESH3D_CLEAN_CAPABILITY]['deadline_seconds'] == 600
    assert stages[MESH3D_CLEAN_CAPABILITY]['weight_ids'] == []
    # Every number carries its source, and the measured scope is recorded with it.
    assert {'min_vram_mb', 'cuda_architecture', 'blender', 'mesh_deadline_seconds',
            'clean_deadline_seconds', 'engine_pins', 'scope', 'determinism',
            'mixed_mode', 'weights'} <= set(description['provenance'])
    assert 'measured' in description['provenance']['mesh_deadline_seconds']
    assert 'front view only' in description['provenance']['scope']
    assert 'not deterministic' in description['provenance']['determinism']
    assert 'engine-v0.2.0' in description['provenance']['engine_pins']
    recipe = description.pop('recipe_version')
    assert recipe == engine_recipe_version(description)


def test_the_fingerprint_covers_pins_and_qualification(monkeypatch):
    qualified = describe_profile(MESH3D)['recipe_version']
    monkeypatch.setitem(PROFILES, MESH3D, unqualified_profile())
    provisional = describe_profile(MESH3D)['recipe_version']
    monkeypatch.setitem(PROFILES, MESH3D, replace(PROFILES[MESH3D], engine=replace(
        MESH3D_RECIPE, engine_commit='d' * 40), qualified=True))
    assert len({provisional, qualified, describe_profile(MESH3D)['recipe_version']}) == 3


def test_an_engine_profile_cannot_be_qualified_with_unset_pins():
    with pytest.raises(ValueError, match='cannot be qualified with unset pins: engine_tag'):
        replace(unqualified_profile(), qualified=True)
    with pytest.raises(ValueError, match='unset pins: weights_lock_sha256'):
        replace(PROFILES[MESH3D], engine=replace(MESH3D_RECIPE, weights_lock_sha256=None))


def test_the_shipped_engine_profile_is_qualified():
    assert PROFILES[MESH3D].qualified and not MESH3D_RECIPE.missing_pins()
    assert machine_profiles.unusable_reason(MESH3D) is None
    assert machine_profiles.unqualified()['profiles'] == frozenset()


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
def test_parse_profiles_refuses_the_unqualified_profile(unqualified, value):
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


async def test_prepare_environment_refuses_before_touching_the_runtime(unqualified, no_runtime):
    with pytest.raises(ProfileNotQualified, match=NOT_QUALIFIED):
        await prepare_environment({'MVS_WORKER_PROFILE': MESH3D})


async def test_prepare_mixed_profiles_refuses_before_touching_the_runtime(unqualified, no_runtime):
    with pytest.raises(ProfileNotQualified, match=NOT_QUALIFIED):
        await agent_main.prepare_mixed_profiles(STATIC_ENVIRONMENT | {'MVS_WORKER_PROFILE': MESH3D})


def test_profile_plugins_refuse_the_unqualified_profile(unqualified):
    from outbound_gpu_worker_pool.engine import AISmithEnginePlugin

    with pytest.raises(ProfileNotQualified, match=NOT_QUALIFIED):
        AISmithEnginePlugin(MESH3D, {RECIPE_KEY: describe_profile(MESH3D)['recipe_version']})
    with pytest.raises(ProfileNotQualified, match=NOT_QUALIFIED):
        machine_profiles.ProfileComfyPlugin(MESH3D, {}, client=None, base_url='http://127.0.0.1:8188')


def test_the_qualified_engine_profile_parses_but_never_shares_a_runtime():
    assert parse_profiles(MESH3D) == [MESH3D]
    with pytest.raises(ValueError, match='cannot share a runtime'):
        parse_profiles(f'{STANDARD},{MESH3D}')


# Coordinator refusals.

def _registration(*capabilities: str, **runtime_versions: str) -> WorkerRegistration:
    return WorkerRegistration(
        worker_id=WORKER_A.worker_id,
        capabilities=tuple(WorkerCapability(c, 'aismith-engine', '1') for c in capabilities),
        runtime_versions=dict(runtime_versions))


@pytest.mark.parametrize('case', ['mesh', 'clean-beside-echo', 'profile-name', 'namespaced-key', 'recipe'])
async def test_registration_advertising_the_unqualified_profile_is_refused(unqualified, case):
    registration = {
        'mesh': _registration(MESH3D_MESH_CAPABILITY),
        'clean-beside-echo': _registration(ECHO, MESH3D_CLEAN_CAPABILITY),
        'profile-name': _registration(ECHO, magic_profile=MESH3D),
        'namespaced-key': _registration(ECHO, **{recipe_key(MESH3D): 'r' * 64}),
        'recipe': _registration(ECHO, magic_recipe=unqualified),
    }[case]
    harness = _harness()
    with pytest.raises(ProfileNotQualified, match='not qualified yet'):
        await harness.service.register_heartbeat(WORKER_A, registration)
    assert await harness.registry.get(WORKER_A.worker_id) is None
    # The same worker without it registers normally.
    await harness.service.register_heartbeat(WORKER_A, _registration(ECHO, python='3.13'))


def test_the_heartbeat_route_answers_422(unqualified):
    from test_routes import _authorization, _harness as routes_harness

    harness = routes_harness()
    response = harness.client.post('/worker/v1/heartbeat', headers=_authorization('token-a'), json={
        'worker_id': 'worker-a', 'tenant_id': 'tenant-a',
        'capabilities': [{'capability_id': MESH3D_MESH_CAPABILITY, 'plugin_id': 'aismith-engine',
                          'plugin_version': '1'}]})
    assert response.status_code == 422
    assert response.json()['detail'] == {'reason': 'profile_not_qualified'}


@pytest.mark.parametrize('case', ['mesh', 'clean', 'recipe', 'namespaced-key', 'profile-name'])
async def test_submit_refuses_jobs_for_the_unqualified_profile(unqualified, case):
    overrides = {
        'mesh': {'capability_id': MESH3D_MESH_CAPABILITY},
        'clean': {'capability_id': MESH3D_CLEAN_CAPABILITY},
        'recipe': {'requirements': JobRequirements(runtime_versions={RECIPE_KEY: unqualified})},
        'namespaced-key': {'requirements': JobRequirements(runtime_versions={recipe_key(MESH3D): 'r' * 64})},
        'profile-name': {'requirements': JobRequirements(runtime_versions={'magic_profile': MESH3D})},
    }[case]
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


async def test_lease_never_hands_out_an_unqualified_job_from_memory(unqualified):
    harness = _harness()
    mesh, echo = await _seeded_around_the_refusals(harness.jobs, harness.registry, assets=harness.assets)
    grant = await harness.service.lease(WORKER_A, (MESH3D_MESH_CAPABILITY, ECHO))
    assert grant.job_id == echo.job_id
    assert await harness.service.lease(WORKER_A, (MESH3D_MESH_CAPABILITY,)) is None
    assert harness.jobs.records[mesh.job_id].status is JobStatus.QUEUED
    assert harness.jobs.records[mesh.job_id].attempts == 0


async def test_a_worker_naming_the_unqualified_profile_leases_nothing_from_memory(unqualified):
    harness = _harness()
    _, echo = await _seeded_around_the_refusals(harness.jobs, harness.registry,
                                                versions={'magic_profile': MESH3D})
    assert await harness.service.lease(WORKER_A, (ECHO,)) is None
    assert harness.jobs.records[echo.job_id].status is JobStatus.QUEUED


async def test_completion_of_an_unqualified_job_is_rejected(unqualified):
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
    # Even the qualified profile: without the registry it cannot be told apart.
    monkeypatch.setitem(sys.modules, 'outbound_gpu_worker_pool.machine_profiles', None)
    harness = _harness()
    with pytest.raises(ProfileNotQualified):
        await harness.service.submit(_submission(capability_id=MESH3D_CLEAN_CAPABILITY))
    with pytest.raises(ProfileNotQualified):
        await harness.service.register_heartbeat(WORKER_A, _registration(ECHO, magic_profile=MESH3D))


# The qualified profile, end to end at the coordinator.

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
async def test_the_coordinator_attests_engine_inputs(capability, payload, effective, accepted):
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


async def test_the_coordinator_rejects_another_engine_configuration():
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


def test_the_engine_recipe_fits_the_heartbeat_schema():
    """Recipe and lock digests stay within the route's runtime-version bounds."""
    from outbound_gpu_worker_pool.routes import WorkerHeartbeatRequest

    WorkerHeartbeatRequest.model_validate({
        'worker_id': 'ilya-desktop',
        'capabilities': [{'capability_id': c, 'plugin_id': 'aismith-engine', 'plugin_version': '1'}
                         for c in PROFILES[MESH3D].capability_ids],
        'runtime_versions': {RECIPE_KEY: describe_profile(MESH3D)['recipe_version'],
                             'aismith_sources_lock': 'b' * 64, 'blender': 'Blender 4.2.23 LTS'},
    })
    assert json.dumps(describe_profile(MESH3D))


def _qualified_versions() -> dict[str, str]:
    return {RECIPE_KEY: describe_profile(MESH3D)['recipe_version'], CONFIG_KEY: 'c' * 64,
            'magic_profile': MESH3D}


async def test_a_registration_advertising_the_qualified_profile_is_accepted():
    harness = _harness()
    record = await harness.service.register_heartbeat(WORKER_A, _registration(
        MESH3D_MESH_CAPABILITY, MESH3D_CLEAN_CAPABILITY, **_qualified_versions()))
    assert set(record.capability_ids) == {MESH3D_MESH_CAPABILITY, MESH3D_CLEAN_CAPABILITY}
    assert record.runtime_versions['magic_profile'] == MESH3D


async def test_submit_accepts_and_lease_hands_out_a_qualified_mesh_job():
    harness = _harness()
    versions = _qualified_versions()
    submission, _ = _engine_job(MESH3D_MESH_CAPABILITY, {'inputs': {'front': 'inputs/pool/front.png'},
                                                        'resolution': 1024, 'seed': 7}, versions[RECIPE_KEY])
    submission = replace(submission, requirements=JobRequirements(
        runtime_versions={RECIPE_KEY: versions[RECIPE_KEY], CONFIG_KEY: versions[CONFIG_KEY]}))
    await harness.assets.write_once('inputs/pool/front.png', b'png', 'image/png')
    record = await harness.service.submit(submission)
    assert record.status is JobStatus.QUEUED
    # A worker on another configuration never gets it; the matching one does.
    await harness.service.register_heartbeat(WORKER_A, _registration(
        MESH3D_MESH_CAPABILITY, **{**versions, CONFIG_KEY: 'd' * 64}))
    assert await harness.service.lease(WORKER_A, (MESH3D_MESH_CAPABILITY,)) is None
    await harness.service.register_heartbeat(WORKER_A, _registration(MESH3D_MESH_CAPABILITY, **versions))
    grant = await harness.service.lease(WORKER_A, (MESH3D_MESH_CAPABILITY,))
    assert grant is not None and grant.job_id == submission.job_id
    assert grant.payload['resolution'] == 1024
