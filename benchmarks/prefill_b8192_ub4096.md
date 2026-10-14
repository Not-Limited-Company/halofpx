# Deep-context probe - prefill_b8192_ub4096

- Needle at 50% of the material; conditions vary the tail only
- Notes: prefill tuning: -b 8192 -ub 4096, ~92K-token code prompt

| content | target tokens | condition | prompt tokens | prefill s | generated | hit cap | retrieved | correct answer |
|---|---:|---|---:|---:|---:|:--:|:--:|:--:|
| code | 92000 | cap1024_plain | 124608 | 1832.2 | 1024 | yes | no | no |
