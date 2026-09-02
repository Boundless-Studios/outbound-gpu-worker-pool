"""Asset store implementations.

Each backend lives in its own module and pulls its own optional dependency, so
importing this package never requires a provider SDK. Install the matching extra
and import the module directly, for example
`from outbound_gpu_worker_pool.assets.gcs import GcsAssetStore` with `[gcs]`.
"""
