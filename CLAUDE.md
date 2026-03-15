# CLAUDE.md

## Rules

- Never cancel a job on a cloud server (Lambda, RunPod, etc.) without getting explicit user permission first. Jobs that look "stale" or "stuck" may actually be important long-running work (e.g. a Gemma finetune that takes 12+ hours). The user may have started training jobs outside of Claude sessions, so don't assume unrecognized jobs are orphaned. Always ask before cancelling — the cost of asking is zero, the cost of killing someone's training run is hours of lost compute.
