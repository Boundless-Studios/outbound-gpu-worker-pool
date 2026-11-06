"""The `aismith-engine` plugin: an engine profile's stages as a subprocess.

A non-ComfyUI `GpuExecutorPlugin`. For each lease it binds the granted inputs
to the stage's spec input names, writes one spec, and runs

    aismith-engine run <stage> --spec spec.json --workdir DIR

in its own process group, bounded by the profile's per-capability deadline
(and the lease's), killing the whole group on cancel or timeout. It reads
`DIR/result.json`: `rejected` is terminal (`PluginRequestRejected`), `failed`
or a crash is a failure carrying the engine's error, and `ok` is one GLB.

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from outbound_gpu_worker_pool import machine_profiles
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
ENGINE_COMMAND_VARIABLE = 'MVS_ENGINE_COMMAND'
# The payload key that maps a stage's spec input names to granted input keys.
INPUTS_PAYLOAD_KEY = 'inputs'
INSPECT_TIMEOUT_SECONDS = 600.0
KILL_GRACE_SECONDS = 10.0
EXIT_STATUSES = {0: 'ok', 1: 'failed', 2: 'rejected'}
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


def engine_versions(profile: str, report: dict[str, Any]) -> dict[str, str]:
    """The runtime versions an engine profile registers, from an `inspect` report.

    Raises EngineDrift when the report does not match the recipe's pins: the
    engine commit and both lock digests, every weight pinned, a GPU of the
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
    if weights.get('ok') is not True or weights.get('pinned') is not True:
        problems.append('weights are not all present and pinned')
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
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
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


async def prepare_engine_environment(profile: str, environment: dict[str, str]) -> dict[str, str]:
    """Enrolment for an engine profile: `inspect --full`, checked against the recipe."""
    machine_profiles.require_qualified(profile)
    command = engine_command(environment)
    report = await run_inspect(command, full=True)
    versions = engine_versions(profile, report)
    gpu = max(_qualifying_gpus(_recipe(profile), report), key=lambda g: g['memory_mb'])
    return {
        'MVS_WORKER_PROFILE': profile,
        ENGINE_COMMAND_VARIABLE: shlex.join(command),
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
                 kill_grace_seconds: float = KILL_GRACE_SECONDS) -> None:
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
        self._running: dict[str, asyncio.subprocess.Process] = {}

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
            report = await run_inspect(self.command, timeout=inspect_timeout,
                                       kill_grace_seconds=self.kill_grace_seconds)
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
        settings = {name: request.inputs[name] for name, _ in stage.settings if name in request.inputs}
        spec_path = context.workspace / 'spec.json'
        spec_path.write_text(json.dumps({
            'schema_version': ENGINE_SPEC_SCHEMA_VERSION, 'stage': stage.stage,
            'inputs': {name: str(context.input_paths[key]) for name, key in bound.items()},
            'settings': settings,
        }, sort_keys=True), encoding='utf-8')
        remaining = (context.deadline - datetime.now(UTC)).total_seconds()
        timeout = min(float(stage.deadline_seconds), remaining)
        if timeout <= 0:
            raise EngineTimedOut('the lease deadline passed before the stage started')
        code = await self._run_stage(
            context, [*self.command, 'run', stage.stage, '--spec', str(spec_path),
                      '--workdir', str(workdir)], timeout)
        result = self._result(workdir, stage, code)
        output = self._output(workdir, result)
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

    async def _run_stage(self, context: ExecutionContext, argv: list[str], timeout: float) -> int:
        logs = context.workspace
        with (logs / 'engine-stdout.log').open('wb') as stdout, \
                (logs / 'engine-stderr.log').open('wb') as stderr:
            process = await asyncio.create_subprocess_exec(
                *argv, stdout=stdout, stderr=stderr, stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True)
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

    def _result(self, workdir: Path, stage: EngineStage, code: int) -> dict[str, Any]:
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
        if EXIT_STATUSES.get(code) != status:
            raise EngineFailed(f'the engine exited {code} with status {status!r}')
        if result.get('stage') != stage.stage or result.get('schema_version') != ENGINE_SPEC_SCHEMA_VERSION:
            raise EngineFailed('the engine result is for another stage or schema')
        error = str(result.get('error') or status)[:MAX_AUDIT_REASON_LENGTH]
        if status == 'rejected':
            raise PluginRequestRejected(error)
        if status == 'failed':
            raise EngineFailed(error)
        if status != 'ok' or code != 0:
            raise EngineFailed(f'the engine exited {code} with status {status!r}')
        engine = result.get('engine') if isinstance(result.get('engine'), dict) else {}
        if (engine.get('commit'), engine.get('sources_lock_sha256'), engine.get('weights_lock_sha256')) != (
                self.versions['aismith_engine_commit'], self.versions['aismith_sources_lock'],
                self.versions['aismith_weights_lock']):
            raise EngineDrift('the engine that ran the stage is not the registered one')
        return result

    def _output(self, workdir: Path, result: dict[str, Any]) -> Path:
        output = result.get('output')
        if not isinstance(output, dict) or not isinstance(output.get('path'), str):
            raise EngineFailed('the engine reported ok without an output')
        path = Path(output['path'])
        path = path if path.is_absolute() else workdir / path
        if path.is_symlink() or not path.resolve().is_relative_to(workdir.resolve()) or not path.is_file():
            raise EngineFailed('the engine output is not a file inside its workdir')
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
        if process is None:
            return False
        await _kill_group(process, self.kill_grace_seconds)
        return True

    async def health(self) -> PluginHealth:
        try:
            actual = engine_versions(self.profile, await run_inspect(
                self.command, timeout=self.inspect_timeout_seconds,
                kill_grace_seconds=self.kill_grace_seconds))
        except (EngineDrift, OSError) as exc:
            return PluginHealth(healthy=False, detail=str(exc)[:MAX_AUDIT_REASON_LENGTH])
        if actual != self.versions:
            return PluginHealth(healthy=False, detail='the engine runtime changed since registration')
        return PluginHealth(healthy=True)
