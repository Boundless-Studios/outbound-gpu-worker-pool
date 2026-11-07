"""The `aismith-engine` plugin: an engine profile's stages, run locally or by a sidecar.

A non-ComfyUI `GpuExecutorPlugin` with two transports:

- `local` (no `OGWP_ENGINE_URL`): for each lease it binds the granted inputs to
  the stage's spec input names, writes one spec, and runs

      aismith-engine run <stage> --spec spec.json --workdir DIR

  in its own process group, killing the whole group on cancel or timeout, and
  reads `DIR/result.json`.
- `http` (`OGWP_ENGINE_URL`, loopback or private only): the hardened worker's
  engine sidecar, `aismith-engine serve`, which holds the GPU and no
  credentials. Per job: `GET /inspect`, `PUT` each input, `POST run`, `GET`
  the output, and always `DELETE` the job; cancel or timeout `POST cancel`.

Either way the stage is bounded by the profile's per-capability deadline (and
the lease's), and the result gets the same checks: `rejected` is terminal
(`PluginRequestRejected`), `failed` or a crash is a failure carrying the
engine's error, and `ok` is one GLB whose magic, size and sha256 are checked
on the bytes the worker holds.

Before each job it runs `aismith-engine inspect` (cached weight hashes) and
fails closed when the engine commit, either lock digest, the weight pinning,
the GPU architecture or the Blender series differ from what the worker
registered at startup (`inspect --full`).

The plugin refuses to exist for an unqualified profile, like enrolment does.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import signal
from contextlib import suppress
from urllib.parse import urlsplit
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from outbound_gpu_worker_pool import machine_profiles
from outbound_gpu_worker_pool.comfy import _require_private_runtime
from outbound_gpu_worker_pool.contracts import (
    MAX_AUDIT_REASON_LENGTH,
    CapabilitySchema,
    JobPayloadValue,
    LeaseGrant,
    WorkerCapability,
)
from outbound_gpu_worker_pool.engine_recipes import (
    AISMITH_ENGINE,
    GLB_CONTENT_TYPE,
    EngineRecipe,
    EngineStage,
)
from outbound_gpu_worker_pool.plugins import (
    CapabilityManifest,
    ExecutionContext,
    PluginHealth,
    PluginOutput,
    PluginRequestRejected,
    ValidatedRequest,
)
from outbound_gpu_worker_pool.validation import validate_job_payload

ENGINE_PLUGIN_ID = 'aismith-engine'
ENGINE_PLUGIN_VERSION = '1'
ENGINE_CONTRACT_VERSION = 1
# The engine's own spec/result schema (`backend/engine/contract.py`).
ENGINE_SPEC_SCHEMA_VERSION = 1
# The worker's engine settings. The runtime roots the engine itself reads
# (TRELLIS_REFINER_DIR, BLENDER_EXE, U2NET_HOME, UNIRIG_DIR, AISMITH_DATA_DIR)
# pass through to it unchanged from the worker's environment.
ENGINE_COMMAND_VARIABLE = 'MVS_ENGINE_COMMAND'
# Set: the `http` transport to an `aismith-engine serve` sidecar. Unset: `local`.
ENGINE_URL_VARIABLE = 'OGWP_ENGINE_URL'
INSPECT_TIMEOUT_VARIABLE = 'MVS_ENGINE_INSPECT_TIMEOUT_SECONDS'
FULL_INSPECT_TIMEOUT_VARIABLE = 'MVS_ENGINE_FULL_INSPECT_TIMEOUT_SECONDS'
KILL_GRACE_VARIABLE = 'MVS_ENGINE_KILL_GRACE_SECONDS'
ENGINE_RUNTIME_ROOTS = ('TRELLIS_REFINER_DIR', 'BLENDER_EXE', 'U2NET_HOME', 'UNIRIG_DIR', 'AISMITH_DATA_DIR')
# The payload key that maps a stage's spec input names to granted input keys.
INPUTS_PAYLOAD_KEY = 'inputs'
INSPECT_TIMEOUT_SECONDS = 600.0
# Enrolment rehashes every weight file (`inspect --full`).
FULL_INSPECT_TIMEOUT_SECONDS = 1800.0
KILL_GRACE_SECONDS = 10.0
EXIT_STATUSES = {0: 'ok', 1: 'failed', 2: 'rejected'}
RESULT_STATUSES = frozenset(EXIT_STATUSES.values())
# The runner's job ids (`aismith-engine serve`).
RUNNER_JOB_ID = re.compile(r'[A-Za-z0-9_-]{1,64}')
# How long a runner request other than `run` and `inspect` may take.
RUNNER_REQUEST_SECONDS = 120.0
# Optional result sections dropped, in this order, when diagnostics would
# exceed the pool's payload bound.
OPTIONAL_DIAGNOSTICS = ('measured', 'gate', 'engine_effective_inputs', 'timings_s', 'weights')


class EngineFailed(RuntimeError):
    """The engine reported `failed`, crashed, or produced an unusable result."""


class EngineTimedOut(EngineFailed):
    """The stage outran its deadline; its process group was killed."""


class EngineCancelled(EngineFailed):
    """The job was cancelled; the stage's process group was killed."""


class EngineDrift(RuntimeError):
    """The engine runtime is not the one this worker registered."""


class InspectTimedOut(EngineDrift):
    """`aismith-engine inspect` did not finish within its bound."""


def engine_command(environment: dict[str, str] | Any) -> tuple[str, ...]:
    raw = environment.get(ENGINE_COMMAND_VARIABLE, '')
    return tuple(shlex.split(raw)) if raw.strip() else (AISMITH_ENGINE,)


def engine_seconds(environment: dict[str, str] | Any, variable: str, default: float) -> float:
    raw = environment.get(variable, '')
    if not raw.strip():
        return default
    value = float(raw)
    if not value > 0:
        raise ValueError(f'{variable} must be a positive number of seconds')
    return value


def _engine_environment() -> dict[str, str]:
    """The worker's environment for the engine, always offline.

    The engine CLI sets AISMITH_OFFLINE=1 itself; the worker sets it too, so
    no launcher environment can turn it off.
    """
    return {**os.environ, 'AISMITH_OFFLINE': '1'}


def _recipe(profile: str) -> EngineRecipe:
    spec = machine_profiles.PROFILES.get(profile)
    if spec is None or spec.engine is None:
        raise ValueError(f'{profile} is not an engine profile')
    return spec.engine


def _qualifying_gpus(recipe: EngineRecipe, report: dict[str, Any]) -> list[dict[str, Any]]:
    gpus = report.get('gpus') if isinstance(report.get('gpus'), list) else []
    return [gpu for gpu in gpus if isinstance(gpu, dict)
            and str(gpu.get('compute_capability')) == recipe.cuda_architecture
            and isinstance(gpu.get('memory_mb'), int) and gpu['memory_mb'] >= recipe.min_vram_mb]


def _section(report: dict[str, Any], *path: str) -> dict[str, Any]:
    value: Any = report
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value if isinstance(value, dict) else {}


def _weight_problems(recipe: Any, weights: dict[str, Any]) -> list[str]:
    """Every file of every weight set the recipe's stages load must verify `ok`.

    Weight sets no stage of this recipe loads (the rig checkpoint, for a
    mesh-and-clean profile) are not required; `unpinned`, `unverified`,
    missing and mismatched files of a required set are all refused.
    """
    files = weights.get('files') if isinstance(weights.get('files'), list) else []
    problems = []
    for weight_id in recipe.weight_ids:
        own = [f for f in files if isinstance(f, dict) and f.get('weight_id') == weight_id]
        if not own:
            problems.append(f'weight set {weight_id} is not reported')
        bad = sorted({str(f.get('status')) for f in own if f.get('status') != 'ok'})
        if bad:
            problems.append(f'weight set {weight_id} is not verified ({", ".join(bad)})')
    return problems


def engine_versions(profile: str, report: dict[str, Any]) -> dict[str, str]:
    """The runtime versions an engine profile registers, from an `inspect` report.

    Raises EngineDrift when the report does not match the recipe's pins: the
    engine commit and both lock digests, every weight its stages load verified, a GPU of the
    recipe's architecture with enough VRAM, the TRELLIS native build's
    architecture, the Blender series, and offline mode.
    """
    recipe = _recipe(profile)
    engine = _section(report, 'engine')
    trellis = _section(report, 'runtimes', 'trellis')
    native = trellis.get('native_contract') if isinstance(trellis.get('native_contract'), dict) else {}
    blender = str(_section(report, 'runtimes', 'blender').get('version') or '')
    weights = _section(report, 'weights')
    problems = [
        f'{label} is {actual!r}, the recipe pins {expected!r}'
        for label, actual, expected in (
            ('engine commit', engine.get('commit'), recipe.engine_commit),
            ('sources lock', engine.get('sources_lock_sha256'), recipe.sources_lock_sha256),
            ('weights lock', engine.get('weights_lock_sha256'), recipe.weights_lock_sha256),
            ('TRELLIS native architecture', native.get('cuda_architecture'), recipe.cuda_architecture),
        )
        if actual is None or actual != expected
    ]
    problems.extend(_weight_problems(recipe, weights))
    if report.get('offline') is not True:
        problems.append('the engine is not offline')
    if not re.fullmatch(rf'Blender {re.escape(recipe.blender_series)}\.\d+( LTS)?', blender):
        problems.append(f'Blender is {blender or "missing"}, the recipe needs {recipe.blender_series}.x LTS')
    gpus = _qualifying_gpus(recipe, report)
    if not gpus:
        problems.append(f'no GPU of compute capability {recipe.cuda_architecture} '
                        f'with {recipe.min_vram_mb} MB')
    if problems:
        raise EngineDrift('; '.join(problems))
    gpu = max(gpus, key=lambda g: g['memory_mb'])
    versions = {
        machine_profiles.RECIPE_KEY: machine_profiles.describe_profile(profile)['recipe_version'],
        'magic_profile': profile,
        'aismith_engine_version': str(engine.get('version')),
        'aismith_engine_commit': str(engine['commit']),
        'aismith_sources_lock': str(engine['sources_lock_sha256']),
        'aismith_weights_lock': str(engine['weights_lock_sha256']),
        'gpu_compute_capability': recipe.cuda_architecture,
        'blender': blender,
        'trellis_torch': str(trellis.get('torch')),
    }
    config = {'versions': versions, 'gpu': gpu.get('name'), 'trellis_cuda': trellis.get('cuda'),
              'native_contract': native, 'comfyui_version': trellis.get('comfyui_version')}
    versions[machine_profiles.CONFIG_KEY] = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    return versions


async def _kill_group(process: asyncio.subprocess.Process, grace_seconds: float) -> None:
    """SIGTERM the process group, SIGKILL whatever is left after the grace."""
    if process.returncode is None:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), grace_seconds)
        except TimeoutError:
            pass
    # Stragglers the leader left behind share its group.
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def run_inspect(command: tuple[str, ...], *, full: bool = False,
                      timeout: float = INSPECT_TIMEOUT_SECONDS,
                      kill_grace_seconds: float = KILL_GRACE_SECONDS) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        *command, 'inspect', *(('--full',) if full else ()),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
        env=_engine_environment())
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        raise InspectTimedOut(f'engine inspect did not finish in {timeout:.0f} s') from None
    finally:
        await _kill_group(process, kill_grace_seconds)
    if process.returncode != 0:
        raise EngineDrift(f'engine inspect exited {process.returncode}')
    try:
        report = json.loads(stdout)
    except ValueError:
        raise EngineDrift('engine inspect printed no JSON report') from None
    if not isinstance(report, dict):
        raise EngineDrift('engine inspect printed no JSON report')
    return report


def runner_url(environment: dict[str, str] | Any) -> str | None:
    """The sidecar runner's base URL, or None for the local transport."""
    raw = environment.get(ENGINE_URL_VARIABLE, '').strip()
    if not raw:
        return None
    if urlsplit(raw).scheme not in ('http', 'https'):
        raise ValueError(f'{ENGINE_URL_VARIABLE} must be an http(s) URL')
    try:
        _require_private_runtime(raw)
    except ValueError:
        raise ValueError(f'{ENGINE_URL_VARIABLE} must address a loopback or private engine runner') from None
    return raw.rstrip('/')


def runner_client() -> httpx.AsyncClient:
    # Never through the worker's egress proxy: the runner is on loopback.
    return httpx.AsyncClient(trust_env=False, follow_redirects=False)


async def http_inspect(client: httpx.AsyncClient, base_url: str, *, full: bool = False,
                       timeout: float = INSPECT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """`GET /inspect` from the sidecar runner: the same report as the CLI."""
    try:
        response = await client.get(f'{base_url}/inspect', params={'full': '1' if full else '0'},
                                    timeout=timeout)
    except httpx.TimeoutException:
        raise InspectTimedOut(f'engine inspect did not finish in {timeout:.0f} s') from None
    except httpx.HTTPError as exc:
        raise EngineDrift(f'the engine runner is unreachable ({type(exc).__name__})') from None
    if response.status_code != 200:
        raise EngineDrift(f'engine inspect answered {response.status_code}')
    try:
        report = response.json()
    except ValueError:
        raise EngineDrift('engine inspect returned no JSON report') from None
    if not isinstance(report, dict):
        raise EngineDrift('engine inspect returned no JSON report')
    return report


async def prepare_engine_environment(profile: str, environment: dict[str, str]) -> dict[str, str]:
    """Enrolment for an engine profile: `inspect --full`, checked against the recipe.

    Over HTTP (`GET /inspect?full=1`) when `OGWP_ENGINE_URL` is set, else the CLI.
    """
    machine_profiles.require_qualified(profile)
    timeout = engine_seconds(environment, FULL_INSPECT_TIMEOUT_VARIABLE, FULL_INSPECT_TIMEOUT_SECONDS)
    url = runner_url(environment)
    if url is not None:
        async with runner_client() as client:
            report = await http_inspect(client, url, full=True, timeout=timeout)
        transport = {ENGINE_URL_VARIABLE: url}
    else:
        command = engine_command(environment)
        report = await run_inspect(command, full=True, timeout=timeout)
        transport = {ENGINE_COMMAND_VARIABLE: shlex.join(command)}
    versions = engine_versions(profile, report)
    gpu = max(_qualifying_gpus(_recipe(profile), report), key=lambda g: g['memory_mb'])
    return {
        'MVS_WORKER_PROFILE': profile,
        **transport,
        'OGWP_WORKER_VRAM_MB': str(gpu['memory_mb']),
        'OGWP_WORKER_GPU_MODEL': str(gpu.get('name')),
        'OGWP_WORKER_RUNTIME_VERSIONS': json.dumps(versions, separators=(',', ':')),
        'OGWP_WORKER_CONCURRENCY': '1',
    }


def _input_schema(stage: EngineStage) -> dict[str, JobPayloadValue]:
    properties: dict[str, JobPayloadValue] = {
        INPUTS_PAYLOAD_KEY: {
            'type': 'object', 'additionalProperties': False,
            'properties': {name: {'type': 'string'} for name in stage.inputs},
            'required': list(stage.required_inputs),
        },
    }
    for name, (kind, values) in stage.settings:
        if kind == 'choices':
            properties[name] = {'enum': list(values)}
        else:
            properties[name] = {'type': 'integer', 'minimum': values[0], 'maximum': values[1]}
    schema: dict[str, JobPayloadValue] = {
        'type': 'object', 'additionalProperties': False, 'properties': properties,
        'required': [INPUTS_PAYLOAD_KEY, *stage.required_settings],
    }
    if stage.one_of_settings:
        schema['oneOf'] = [{'required': [name]} for name in stage.one_of_settings]
    return schema


def _checked_setting(name: str, kind: str, values: tuple[Any, ...], value: Any) -> JobPayloadValue:
    if kind == 'choices':
        if isinstance(value, bool) or value not in values:
            raise PluginRequestRejected(f'{name} must be one of {", ".join(map(str, values))}')
        return value
    if isinstance(value, bool) or not isinstance(value, int) or not values[0] <= value <= values[1]:
        raise PluginRequestRejected(f'{name} must be an integer from {values[0]} to {values[1]}')
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Diagnostics within the pool's payload bound; optional sections give way first."""
    diagnostics = json.loads(json.dumps(diagnostics, default=str))
    for section in (None, *OPTIONAL_DIAGNOSTICS):
        if section is not None and section in diagnostics:
            encoded = json.dumps(diagnostics[section], sort_keys=True).encode()
            diagnostics[section] = {'omitted': 'exceeds the diagnostics bound',
                                    'sha256': hashlib.sha256(encoded).hexdigest()}
        try:
            validate_job_payload(diagnostics)
            return diagnostics
        except ValueError:
            continue
    raise EngineFailed('engine diagnostics exceed the pool payload bound')


class AISmithEnginePlugin:
    """Serve an engine profile's capabilities through the pinned `aismith-engine`."""

    def __init__(self, profile: str, versions: dict[str, str], *,
                 command: tuple[str, ...] = (AISMITH_ENGINE,), namespaced: bool = False,
                 inspect_timeout_seconds: float = INSPECT_TIMEOUT_SECONDS,
                 kill_grace_seconds: float = KILL_GRACE_SECONDS,
                 engine_url: str | None = None,
                 http_client: httpx.AsyncClient | None = None) -> None:
        machine_profiles.require_qualified(profile)
        self.recipe = _recipe(profile)
        recipe_version = machine_profiles.describe_profile(profile)['recipe_version']
        if versions.get(machine_profiles.RECIPE_KEY) != recipe_version:
            raise ValueError(f'the registered versions are not the {profile} recipe')
        self.profile = profile
        self.versions = dict(versions)
        self.command = tuple(command)
        # A further profile of a multi-profile runtime attests under its
        # namespaced keys, which are what its jobs pin.
        self.namespaced = namespaced
        self.inspect_timeout_seconds = inspect_timeout_seconds
        self.kill_grace_seconds = kill_grace_seconds
        # `http` when a sidecar runner URL is given, else `local`.
        self.engine_url = runner_url({ENGINE_URL_VARIABLE: engine_url}) if engine_url else None
        self.transport = 'http' if self.engine_url else 'local'
        self._client = (http_client or runner_client()) if self.engine_url else None
        self._running: dict[str, asyncio.subprocess.Process] = {}
        self._cancels: dict[str, asyncio.Event] = {}

    def capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            plugin_id=ENGINE_PLUGIN_ID,
            plugin_version=ENGINE_PLUGIN_VERSION,
            capabilities=tuple(
                WorkerCapability(capability_id=stage.capability_id, plugin_id=ENGINE_PLUGIN_ID,
                                 plugin_version=ENGINE_PLUGIN_VERSION)
                for stage in self.recipe.stages),
            schemas=tuple(
                CapabilitySchema(capability_id=stage.capability_id,
                                 contract_version=ENGINE_CONTRACT_VERSION,
                                 input_schema=_input_schema(stage))
                for stage in self.recipe.stages),
        )

    def validate(self, lease: LeaseGrant) -> ValidatedRequest:
        stage = self.recipe.stage_for(lease.capability_id)
        if stage is None:
            raise PluginRequestRejected('unsupported capability')
        if lease.contract_version != ENGINE_CONTRACT_VERSION:
            raise PluginRequestRejected('unsupported contract version')
        domains = dict(stage.settings)
        if set(lease.payload) - set(domains) - {INPUTS_PAYLOAD_KEY}:
            raise PluginRequestRejected('payload carries unsupported keys')
        bound = lease.payload.get(INPUTS_PAYLOAD_KEY, {})
        if not isinstance(bound, dict):
            raise PluginRequestRejected(f'{INPUTS_PAYLOAD_KEY} must map a stage input to a granted key')
        if set(bound) - set(stage.inputs):
            raise PluginRequestRejected(f'{INPUTS_PAYLOAD_KEY} names an input the stage does not take')
        for name in stage.required_inputs:
            if name not in bound:
                raise PluginRequestRejected(f'input {name} is required')
        for name, key in bound.items():
            if not isinstance(key, str) or key not in lease.input_keys:
                raise PluginRequestRejected(f'input {name} must bind a granted input')
        if set(lease.input_keys) - set(bound.values()):
            raise PluginRequestRejected('every granted input must be bound to a stage input')
        inputs: dict[str, JobPayloadValue] = {}
        for name, (kind, values) in stage.settings:
            if name in lease.payload:
                inputs[name] = _checked_setting(name, kind, values, lease.payload[name])
        for name in stage.required_settings:
            if name not in inputs:
                raise PluginRequestRejected(f'{name} is required')
        if stage.one_of_settings and sum(name in inputs for name in stage.one_of_settings) != 1:
            raise PluginRequestRejected(f'exactly one of {", ".join(stage.one_of_settings)} is required')
        seed = inputs.get('seed')
        inputs[INPUTS_PAYLOAD_KEY] = dict(bound)
        return ValidatedRequest(capability_id=lease.capability_id,
                                contract_version=lease.contract_version, inputs=inputs,
                                seed=seed if isinstance(seed, int) else None)

    async def _inspect(self, *, full: bool = False, timeout: float) -> dict[str, Any]:
        if self.engine_url is not None:
            assert self._client is not None
            return await http_inspect(self._client, self.engine_url, full=full, timeout=timeout)
        return await run_inspect(self.command, full=full, timeout=timeout,
                                 kill_grace_seconds=self.kill_grace_seconds)

    async def execute(self, context: ExecutionContext, request: ValidatedRequest) -> PluginOutput:
        stage = self.recipe.stage_for(request.capability_id)
        if stage is None:
            raise PluginRequestRejected('unsupported capability')
        # The pre-job inspection spends the job's own execution time: an expired
        # context starts no process, and inspect never outlives the deadline.
        remaining = (context.deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise EngineTimedOut('the lease deadline passed before the engine was inspected')
        inspect_timeout = min(remaining, self.inspect_timeout_seconds)
        try:
            report = await self._inspect(timeout=inspect_timeout)
        except InspectTimedOut:
            if inspect_timeout < self.inspect_timeout_seconds:
                raise EngineTimedOut('the lease deadline passed while the engine was inspected') from None
            raise
        actual = engine_versions(self.profile, report)
        if actual != self.versions:
            raise EngineDrift('the engine runtime changed; restart the worker and resolve a new recipe')
        workdir = context.workspace / 'engine'
        workdir.mkdir(parents=True)
        bound = request.inputs[INPUTS_PAYLOAD_KEY]
        assert isinstance(bound, dict)
        inputs = {name: context.input_paths[key] for name, key in bound.items()}
        settings = {name: request.inputs[name] for name, _ in stage.settings if name in request.inputs}
        remaining = (context.deadline - datetime.now(UTC)).total_seconds()
        timeout = min(float(stage.deadline_seconds), remaining)
        if timeout <= 0:
            raise EngineTimedOut('the lease deadline passed before the stage started')
        cancel = self._cancels.setdefault(context.job_id, asyncio.Event())
        try:
            if self.engine_url is not None:
                result, output = await self._run_http(context, stage, inputs, settings, timeout,
                                                      workdir, cancel)
            else:
                result, output = await self._run_local(context, stage, inputs, settings, timeout,
                                                       workdir)
        finally:
            self._cancels.pop(context.job_id, None)
        await context.progress(100)
        effective = result.get('effective_inputs')
        effective = effective if isinstance(effective, dict) else {}
        weights = result.get('weights') if isinstance(result.get('weights'), dict) else None
        attested = (machine_profiles.attested_versions(self.profile, actual)
                    if self.namespaced else dict(actual))
        diagnostics = {
            'runtime_versions': attested,
            # Only the attested inputs this job set, as the engine reports it ran
            # them; the coordinator compares these with the job's payload.
            'effective_inputs': {name: effective.get(name) for name in stage.attested_inputs
                                 if name in request.inputs},
            'engine_effective_inputs': effective,
            'stage': stage.stage,
            'engine': result.get('engine'),
            'gate': result.get('gate'),
            'measured': result.get('measured'),
            'weights': None if weights is None else {'ok': weights.get('ok'),
                                                     'pinned': weights.get('pinned')},
            'timings_s': result.get('timings_s'),
        }
        return PluginOutput(path=output, content_type=GLB_CONTENT_TYPE,
                            model_id=f'{AISMITH_ENGINE}/{stage.stage}',
                            model_version=self.versions['aismith_engine_commit'],
                            seed=request.seed, diagnostics=_bounded(diagnostics))

    # The local transport: the CLI as a subprocess.

    async def _run_local(self, context: ExecutionContext, stage: EngineStage, inputs: dict[str, Path],
                         settings: dict[str, Any], timeout: float,
                         workdir: Path) -> tuple[dict[str, Any], Path]:
        spec_path = context.workspace / 'spec.json'
        spec_path.write_text(json.dumps({
            'schema_version': ENGINE_SPEC_SCHEMA_VERSION, 'stage': stage.stage,
            'inputs': {name: str(path) for name, path in inputs.items()},
            'settings': settings,
        }, sort_keys=True), encoding='utf-8')
        code = await self._run_stage(
            context, [*self.command, 'run', stage.stage, '--spec', str(spec_path),
                      '--workdir', str(workdir)], timeout)
        # A crash or signal is never an engine status, whatever result.json says.
        if code not in EXIT_STATUSES:
            raise EngineFailed(f'the engine exited {code}, which is not an engine status code')
        try:
            result = json.loads((workdir / 'result.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise EngineFailed(f'the engine exited {code} without a readable result.json') from None
        if not isinstance(result, dict):
            raise EngineFailed('the engine result is not a JSON object')
        status = result.get('status')
        if EXIT_STATUSES[code] != status:
            raise EngineFailed(f'the engine exited {code} with status {status!r}')
        self._checked_result(result, stage)
        output = result.get('output')
        if not isinstance(output, dict) or not isinstance(output.get('path'), str):
            raise EngineFailed('the engine reported ok without an output')
        path = Path(output['path'])
        path = path if path.is_absolute() else workdir / path
        if path.is_symlink() or not path.resolve().is_relative_to(workdir.resolve()) or not path.is_file():
            raise EngineFailed('the engine output is not a file inside its workdir')
        return result, self._checked_output(path, output)

    async def _run_stage(self, context: ExecutionContext, argv: list[str], timeout: float) -> int:
        logs = context.workspace
        with (logs / 'engine-stdout.log').open('wb') as stdout, \
                (logs / 'engine-stderr.log').open('wb') as stderr:
            process = await asyncio.create_subprocess_exec(
                *argv, stdout=stdout, stderr=stderr, stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True, env=_engine_environment())
        self._running[context.job_id] = process
        waiter = asyncio.ensure_future(process.wait())
        cancelled = asyncio.ensure_future(context.cancel.wait())
        try:
            done, _ = await asyncio.wait({waiter, cancelled}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if waiter in done:
                return waiter.result()
            if cancelled in done:
                raise EngineCancelled('the job was cancelled; the engine process group was killed')
            raise EngineTimedOut(f'the engine exceeded its {timeout:.0f} s deadline; '
                                 'its process group was killed')
        finally:
            cancelled.cancel()
            # Also on our own cancellation: never leave a stage running.
            await _kill_group(process, self.kill_grace_seconds)
            waiter.cancel()
            self._running.pop(context.job_id, None)

    # The http transport: the `aismith-engine serve` sidecar.

    async def _runner(self, method: str, path: str, *, timeout: float = RUNNER_REQUEST_SECONDS,
                      **kwargs: Any) -> httpx.Response:
        assert self._client is not None and self.engine_url is not None
        try:
            return await self._client.request(method, f'{self.engine_url}{path}', timeout=timeout, **kwargs)
        except httpx.TimeoutException:
            raise EngineTimedOut(f'the engine runner did not answer {method} {path} in time') from None
        except httpx.HTTPError as exc:
            raise EngineFailed(f'the engine runner failed {method} {path} ({type(exc).__name__})') from None

    async def _run_http(self, context: ExecutionContext, stage: EngineStage, inputs: dict[str, Path],
                        settings: dict[str, Any], timeout: float, workdir: Path,
                        cancel: asyncio.Event) -> tuple[dict[str, Any], Path]:
        job = context.job_id
        if not RUNNER_JOB_ID.fullmatch(job):
            raise EngineFailed('the job id is not a valid engine runner job id')
        started = asyncio.get_running_loop().time()

        def left() -> float:
            return timeout - (asyncio.get_running_loop().time() - started)

        try:
            for name, path in inputs.items():
                if left() <= 0:
                    raise EngineTimedOut('the stage deadline passed while uploading inputs')
                response = await self._runner('PUT', f'/jobs/{job}/inputs/{name}',
                                              content=path.read_bytes(), timeout=left())
                if response.status_code != 204:
                    raise EngineFailed(f'the engine runner refused input {name} ({response.status_code})')
            result = await self._http_stage(context, stage, inputs, settings, left(), cancel)
            self._checked_result(result, stage)
            output = result.get('output')
            if not isinstance(output, dict) or output.get('url') != f'/jobs/{job}/output':
                raise EngineFailed('the engine runner reported ok without its output URL')
            if left() <= 0:
                raise EngineTimedOut('the stage deadline passed before the output was fetched')
            path = await self._fetch_output(job, output, workdir / f'{stage.stage}.glb', left())
            return result, self._checked_output(path, output)
        finally:
            # Always: the runner deletes the job's work directory (and cancels it if running).
            with suppress(EngineFailed):
                await self._runner('DELETE', f'/jobs/{job}')

    async def _http_stage(self, context: ExecutionContext, stage: EngineStage, inputs: dict[str, Path],
                          settings: dict[str, Any], timeout: float,
                          cancel: asyncio.Event) -> dict[str, Any]:
        job = context.job_id
        if timeout <= 0:
            raise EngineTimedOut('the stage deadline passed before the stage started')
        # The runner enforces the deadline too; the client keeps its own.
        run = asyncio.ensure_future(self._runner('POST', f'/jobs/{job}/run', timeout=timeout + 30, json={
            'schema_version': ENGINE_SPEC_SCHEMA_VERSION, 'stage': stage.stage,
            'inputs': sorted(inputs), 'settings': settings,
            'deadline_seconds': max(1, int(timeout)),
        }))
        waiters = {asyncio.ensure_future(context.cancel.wait()), asyncio.ensure_future(cancel.wait())}
        try:
            done, _ = await asyncio.wait({run, *waiters}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if run not in done:
                if done:
                    raise EngineCancelled('the job was cancelled; the engine runner cancelled it')
                raise EngineTimedOut(f'the engine exceeded its {timeout:.0f} s deadline; '
                                     'the engine runner cancelled it')
            response = run.result()
        finally:
            for waiter in waiters:
                waiter.cancel()
            if not run.done():
                run.cancel()
                # Also on our own cancellation: never leave a stage running.
                with suppress(EngineFailed, asyncio.CancelledError):
                    await self._runner('POST', f'/jobs/{job}/cancel')
        if response.status_code != 200:
            raise EngineFailed(f'the engine runner answered {response.status_code} to run')
        try:
            result = response.json()
        except ValueError:
            raise EngineFailed('the engine runner returned no result JSON') from None
        if not isinstance(result, dict):
            raise EngineFailed('the engine result is not a JSON object')
        if result.get('status') not in RESULT_STATUSES:
            raise EngineFailed(f'the engine runner returned status {result.get("status")!r}')
        return result

    async def _fetch_output(self, job: str, output: dict[str, Any], destination: Path,
                            timeout: float) -> Path:
        expected = output.get('bytes')
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise EngineFailed('the engine result carries no output size')
        assert self._client is not None and self.engine_url is not None
        written = 0
        try:
            async with self._client.stream('GET', f'{self.engine_url}/jobs/{job}/output',
                                           timeout=timeout) as response:
                if response.status_code != 200:
                    raise EngineFailed(f'the engine runner answered {response.status_code} for the output')
                if response.headers.get('X-Sha256') != output.get('sha256'):
                    raise EngineFailed('the engine output X-Sha256 does not match the result')
                if response.headers.get('Content-Length') != str(expected):
                    raise EngineFailed('the engine output Content-Length does not match the result')
                with destination.open('wb') as handle:
                    async for chunk in response.aiter_bytes():
                        written += len(chunk)
                        if written > expected:
                            raise EngineFailed('the engine output is larger than the result says')
                        handle.write(chunk)
        except httpx.TimeoutException:
            raise EngineTimedOut('the engine output did not arrive before the deadline') from None
        except httpx.HTTPError as exc:
            raise EngineFailed(f'the engine output could not be fetched ({type(exc).__name__})') from None
        return destination

    # Checks both transports share.

    def _checked_result(self, result: dict[str, Any], stage: EngineStage) -> None:
        status = result.get('status')
        if result.get('stage') != stage.stage or result.get('schema_version') != ENGINE_SPEC_SCHEMA_VERSION:
            raise EngineFailed('the engine result is for another stage or schema')
        error = str(result.get('error') or status)[:MAX_AUDIT_REASON_LENGTH]
        if status == 'rejected':
            raise PluginRequestRejected(error)
        if status == 'failed':
            raise EngineFailed(error)
        if status != 'ok':
            raise EngineFailed(f'the engine returned status {status!r}')
        engine = result.get('engine') if isinstance(result.get('engine'), dict) else {}
        if (engine.get('commit'), engine.get('sources_lock_sha256'), engine.get('weights_lock_sha256')) != (
                self.versions['aismith_engine_commit'], self.versions['aismith_sources_lock'],
                self.versions['aismith_weights_lock']):
            raise EngineDrift('the engine that ran the stage is not the registered one')

    @staticmethod
    def _checked_output(path: Path, output: dict[str, Any]) -> Path:
        """Magic, size and sha256, on the bytes the worker holds."""
        if output.get('content_type') != GLB_CONTENT_TYPE:
            raise EngineFailed(f'the engine output is {output.get("content_type")!r}, not a GLB')
        with path.open('rb') as handle:
            if handle.read(4) != b'glTF':
                raise EngineFailed('the engine output is not a binary glTF')
        if path.stat().st_size != output.get('bytes') or _sha256(path) != output.get('sha256'):
            raise EngineFailed('the engine output does not match its reported size and sha256')
        return path

    async def cancel(self, job_id: str) -> bool:
        process = self._running.get(job_id)
        if process is not None:
            await _kill_group(process, self.kill_grace_seconds)
            return True
        event = self._cancels.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    async def health(self) -> PluginHealth:
        try:
            actual = engine_versions(self.profile, await self._inspect(timeout=self.inspect_timeout_seconds))
        except (EngineDrift, OSError) as exc:
            return PluginHealth(healthy=False, detail=str(exc)[:MAX_AUDIT_REASON_LENGTH])
        if actual != self.versions:
            return PluginHealth(healthy=False, detail='the engine runtime changed since registration')
        return PluginHealth(healthy=True)


def engine_plugin_from_env(profile: str, environment: dict[str, str] | Any) -> AISmithEnginePlugin:
    """The plugin for a prepared engine environment (`prepare_engine_environment`).

    `OGWP_ENGINE_URL` set selects the `http` transport, else `local`.
    """
    return AISmithEnginePlugin(
        profile, json.loads(environment['OGWP_WORKER_RUNTIME_VERSIONS']),
        command=engine_command(environment), engine_url=runner_url(environment),
        inspect_timeout_seconds=engine_seconds(environment, INSPECT_TIMEOUT_VARIABLE, INSPECT_TIMEOUT_SECONDS),
        kill_grace_seconds=engine_seconds(environment, KILL_GRACE_VARIABLE, KILL_GRACE_SECONDS))
