# Deep-context probe - corrected_no_reasoning

- Needle at 50% of the material; conditions vary the tail only
- Notes: corrected: nonsense needle, uncontaminated corpora, reasoning off

| content | target tokens | condition | prompt tokens | prefill s | generated | hit cap | retrieved | correct answer |
|---|---:|---|---:|---:|---:|:--:|:--:|:--:|
| wikitext | 46000 | cap1024_plain | 42308 | 326.1 | 11 | no | no | no |
| wikitext | 46000 | cap1024_strict | 42313 | 11.4 | 5 | no | no | no |
| wikitext | 46000 | cap4096_plain | 42308 | 11.0 | 11 | no | no | no |
| wikitext | 46000 | cap4096_strict | 42313 | 11.2 | 5 | no | no | no |
| wikitext | 92000 | cap1024_plain | 84208 | 908.9 | 17 | no | yes | yes |
| wikitext | 92000 | cap1024_strict | 84213 | 17.4 | 16 | no | yes | yes |
| wikitext | 92000 | cap4096_plain | 84208 | 17.3 | 17 | no | yes | yes |
| wikitext | 92000 | cap4096_strict | 84213 | 17.1 | 5 | no | no | no |
| code | 46000 | cap1024_plain | 68573 | 671.5 | 17 | no | no | no |
| code | 46000 | cap1024_strict | 68578 | 15.7 | 1024 | yes | no | no |
| code | 46000 | cap4096_plain | 68573 | 15.2 | 17 | no | no | no |
| code | 46000 | cap4096_strict | 68578 | 15.7 | 37 | no | no | no |
| code | 92000 | cap1024_plain | 102834 | 1264.0 | 1024 | yes | no | no |
| code | 92000 | cap1024_strict | 102839 | 20.1 | 1024 | yes | no | no |
| code | 92000 | cap4096_plain | 102834 | 19.9 | 4096 | yes | no | no |
| code | 92000 | cap4096_strict | 102839 | 20.0 | 4096 | yes | no | no |
