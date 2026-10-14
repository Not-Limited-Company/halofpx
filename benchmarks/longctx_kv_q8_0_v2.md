# Long-context probe — kv_q8_0_v2

- Retrieval: **5/9** (56%)
- Timestamp 20261013_120250 | engine http://127.0.0.1:8800 | max_tokens 1024
- Notes: IQ2_XXS + DSpark, ROCm0, ctx 262144, KV q8_0; varied filler; needle at 25/50/75%; max_tokens 1024 (no truncation); engine port 8800

| depth (tokens) | needle at | repeat | prompt tokens | prefill s | prefill tok/s | decode tok/s | found |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 16384 | 25% | 0 | 23179 | 0.2 | 130747.2 | 13.91 | yes (answer) |
| 16384 | 50% | 0 | 23179 | 0.2 | 140784.2 | 14.26 | yes (answer) |
| 16384 | 75% | 0 | 23179 | 0.2 | 140824.4 | 13.55 | no |
| 32768 | 25% | 0 | 46469 | 0.2 | 254634.4 | 16.62 | yes (answer) |
| 32768 | 50% | 0 | 46469 | 0.2 | 255196.6 | 11.31 | no |
| 32768 | 75% | 0 | 46469 | 0.2 | 255740.1 | 13.89 | yes (answer) |
| 65536 | 25% | 0 | 92636 | 0.2 | 432145.5 | 17.49 | no |
| 65536 | 50% | 0 | 92636 | 0.2 | 430939.3 | 14.28 | no |
| 65536 | 75% | 0 | 92636 | 0.2 | 430546.7 | 9.63 | yes |
