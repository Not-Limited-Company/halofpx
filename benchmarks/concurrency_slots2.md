# Concurrency probe - slots2_ctx524288

- 2 simultaneous requests, max_tokens 256, wall 34.8 s
- Mean per-stream decode: **8.9 tok/s** | aggregate: **14.69 tok/s**
- Notes: 2 slots, 524288 pool (~262144 each), short prompts

| stream | ok | generated | decode tok/s | TTFT ms | wall s |
|---:|:--:|---:|---:|---:|---:|
| 0 | yes | 256 | 7.59 | 1173.8 | 34.8 |
| 1 | yes | 256 | 10.21 | 1159.0 | 26.2 |
