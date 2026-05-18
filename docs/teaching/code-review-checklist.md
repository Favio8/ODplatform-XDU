# Code Review Checklist

- Verify module boundaries stay aligned with the layered architecture
- Keep CLI, service, core, and common responsibilities separated
- Avoid committing datasets, checkpoints, or runtime outputs
- Preserve idempotency for repository setup scripts
