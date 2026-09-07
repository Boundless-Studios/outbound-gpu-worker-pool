# Security policy

This alpha library runs approved work on machines the operator already trusts. It is not a
sandbox for hostile plugins or a marketplace for anonymous GPU providers. An enrolled worker
can read the exact assets granted to its jobs and can return incorrect generated content;
output hashes verify publication integrity, not model honesty.

## Reporting

Do not post credentials, signed asset URLs, private customer inputs, or an operational exploit
against a live service in a public issue. Use the repository's **Security → Report a
vulnerability** option when private vulnerability reporting is enabled. Otherwise open a
content-free issue requesting a private reporting channel; do not include exploit details until
a maintainer has provided one. Native GitHub reporting/secret-protection settings require a
repository administrator to verify; this source change does not assert those settings are on.

## Deployment boundaries

Authenticate workers and approve their full identity and tenant on the coordinator. Keep
Google auto-enrollment off unless an explicit full-identity enrollment allowlist and narrowly
scoped external IAM admission are both configured. A name prefix, client-supplied tenant, or
valid Google token alone is never an admission policy. Existing worker identity and tenant
bindings cannot be replaced by a heartbeat. Keep credentials per machine and revoke compromised
workers; rotating a credential does not itself undo revocation.

Product job submission, asset ownership checks, quotas, and administrator authorization remain
the host application's responsibility. The packaged pool-status router requires explicit
administrator authorization. Tenant-facing applications should implement separate routes that
derive tenant scope from their authenticated principal, never from a query parameter.

Use HTTPS and a trusted ingress. Apply deployment-wide rate, concurrency, and request-body
limits before the application. In-process admission limits are bounded defense in depth, not a
DDoS service; multiple processes each have their own budget. Configure proxy-header trust only
for known proxies and never trust arbitrary Internet `X-Forwarded-For` values.

## Secrets prevention and response

The `Secrets / secrets` CI job runs checksum-pinned Gitleaks against fetched Git history and raw
reachable blob/commit objects. It fails closed on findings or incomplete scans, ignores inline
`gitleaks:allow` comments, and never prints raw scanner matches or credentials. Reports live only
in a temporary directory and are not uploaded as artifacts. `.gitignore` excludes documented
local credential filenames while keeping example files trackable; it is not a security boundary
and cannot remove an already tracked credential.

An administrator should enable and verify GitHub secret scanning, push protection, and private
vulnerability reporting where supported, review existing alerts, and make CI checks required.
Review changes to security workflows and scanner code as security-sensitive changes. Do not add
blanket allowlists for tests or documentation. Any future fixture exception must be narrow,
reviewed, and documented.

If a real credential was committed, **revoke or rotate it immediately** and investigate its
permissions and use. Deleting a file or rewriting history does not revoke a credential and cannot
remove copies already fetched by others. Review branches, retained pull-request refs, workflow
logs/artifacts, releases, and connected deployments. This CI does not cover unreachable server
objects, external LFS payloads, or historical Actions logs/artifacts. A zero-finding scan is not
a guarantee that no unknown secret format exists.
