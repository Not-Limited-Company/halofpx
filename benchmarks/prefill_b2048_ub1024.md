# Deep-context probe - prefill_b2048_ub1024

- Needle at 50% of the material; conditions vary the tail only
- Notes: prefill tuning: -b 2048 -ub 1024, ~92K-token code prompt

| content | target tokens | condition | prompt tokens | prefill s | generated | hit cap | retrieved | correct answer |
|---|---:|---|---:|---:|---:|:--:|:--:|:--:|
| code | 92000 | cap1024_plain | 124608 | 1730.2 | 1024 | yes | no | no |
