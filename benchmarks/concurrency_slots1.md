# Concurrency probe - slots1_ctx1048576

- 1 simultaneous requests, max_tokens 256, wall 15.8 s
- Mean per-stream decode: **16.94 tok/s** | aggregate: **16.21 tok/s**
- Notes: single stream baseline at ctx 1048576

| stream | ok | generated | decode tok/s | TTFT ms | wall s |
|---:|:--:|---:|---:|---:|---:|
| 0 | yes | 256 | 16.94 | 705.4 | 15.8 |
