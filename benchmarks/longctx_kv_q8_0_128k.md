# Long-context probe — kv_q8_0_128k

- Retrieval: **0/1** (0%)
- Timestamp 20261013_105119 | engine http://127.0.0.1:8800 | max_tokens 192
- Notes: IQ2_XXS + DSpark, ROCm0, ctx 262144, KV q8_0; varied filler; needle at 25/50/75% depth; engine port 8800; max_tokens 192 (single 128K probe)

| depth (tokens) | needle at | repeat | prompt tokens | prefill s | prefill tok/s | decode tok/s | found |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 131072 | 50% | 0 | 185621 | 3317.3 | 56.0 | 8.87 | no |
