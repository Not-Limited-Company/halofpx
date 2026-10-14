# Deep-context probe - prefill_b4096_ub2048

- Needle at 50% of the material; conditions vary the tail only
- Notes: prefill tuning: -b 4096 -ub 2048, ~92K-token code prompt

| content | target tokens | condition | prompt tokens | prefill s | generated | hit cap | retrieved | correct answer |
|---|---:|---|---:|---:|---:|:--:|:--:|:--:|
| code | 92000 | cap1024_plain | 124608 | 1693.9 | 1024 | yes | no | no |
