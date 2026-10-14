# Long-context probe — kv_q4_0

- Retrieval: **5/9** (56%)
- Timestamp 20261013_181403 | engine http://127.0.0.1:8800 | max_tokens 1024
- Notes: IQ2_XXS, ROCm0, ctx 262144, KV q4_0, reasoning budget 512, max_tokens 1024

| depth (tokens) | needle at | repeat | prompt tokens | prefill s | prefill tok/s | decode tok/s | found |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 16384 | 25% | 0 | 23179 | 138.4 | 167.4 | 15.16 | yes (answer) |
| 16384 | 50% | 0 | 23179 | 142.4 | 162.8 | 16.11 | yes (answer) |
| 16384 | 75% | 0 | 23179 | 143.0 | 162.0 | 10.09 | no |
| 32768 | 25% | 0 | 46469 | 360.0 | 129.1 | 20.48 | yes (answer) |
| 32768 | 50% | 0 | 46469 | 358.7 | 129.6 | 18.53 | no |
| 32768 | 75% | 0 | 46469 | 359.4 | 129.3 | 12.38 | yes (answer) |
| 65536 | 25% | 0 | 92636 | 1023.5 | 90.5 | 14.77 | no |
| 65536 | 50% | 0 | 92636 | 1022.4 | 90.6 | 10.22 | yes (answer) |
| 65536 | 75% | 0 | 92636 | 1023.1 | 90.5 | 17.46 | no |
