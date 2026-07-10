# CLAUDE.md

## Rules

- Cloud jobs (Lambda, RunPod, etc.):
  - **When working autonomously** (the user has set you to work on your own and stepped away): you may cancel/restart/manage cloud jobs on your own judgment — no need to ask first. In exchange, you MUST set up automation to check on the work: wake yourself every ~30 minutes while things are getting started (spin-up, first steps), then every ~2 hours once it's cooking steadily, to verify status and results. A silent failure (e.g. a job that's "running" but pinned to CPU) must not go hours unnoticed — a health check should catch a stall within one interval, not when the user next asks.
  - **When the user is interactively present**: do NOT cancel a job without explicit permission. Jobs that look "stale" or "stuck" may be important long-running work (e.g. a 12+ hour Gemma finetune), and the user may have started jobs outside Claude sessions — don't assume unrecognized jobs are orphaned. Ask first; the cost of asking is zero.
