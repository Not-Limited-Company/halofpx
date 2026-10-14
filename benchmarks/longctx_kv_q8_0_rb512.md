# Long-context probe — kv_q8_0_rb512

- Retrieval: **5/9** (56%)
- Timestamp 20261013_193542 | engine http://127.0.0.1:8800 | max_tokens 1024
- Notes: IQ2_XXS, ROCm0, ctx 262144, KV q8_0, reasoning budget 512, max_tokens 1024

| depth (tokens) | needle at | repeat | prompt tokens | prefill s | prefill tok/s | decode tok/s | found |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 16384 | 25% | 0 | 23179 | 138.8 | 167.0 | 14.8 | yes (answer) |
| 16384 | 50% | 0 | 23179 | 142.7 | 162.4 | 13.87 | yes (answer) |
| 16384 | 75% | 0 | 23179 | 143.2 | 161.8 | 12.53 | no |
| 32768 | 25% | 0 | 46469 | 360.7 | 128.8 | 14.51 | yes (answer) |
| 32768 | 50% | 0 | 46469 | 360.3 | 129.0 | 15.58 | no |
| 32768 | 75% | 0 | 46469 | 360.1 | 129.0 | 16.18 | yes (answer) |
| 65536 | 25% | 0 | 92636 | 1025.0 | 90.4 | 11.68 | no |
| 65536 | 50% | 0 | 92636 | 1026.0 | 90.3 | 16.03 | no |
| 65536 | 75% | 0 | 92636 | 1026.7 | 90.2 | 9.92 | yes (answer) |
