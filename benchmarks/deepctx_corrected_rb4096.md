# Deep-context probe - corrected_rb4096

- Needle at 50% of the material; conditions vary the tail only
- Notes: corrected: nonsense needle, uncontaminated corpora, reasoning budget 4096

| content | target tokens | condition | prompt tokens | prefill s | generated | hit cap | retrieved | correct answer |
|---|---:|---|---:|---:|---:|:--:|:--:|:--:|
| wikitext | 46000 | cap1024_plain | 42308 | 321.0 | 1024 | yes | no | no |
| wikitext | 46000 | cap1024_strict | 42313 | 11.4 | 46 | no | no | no |
| wikitext | 46000 | cap4096_plain | 42308 | 11.0 | 4096 | yes | no | no |
| wikitext | 46000 | cap4096_strict | 42313 | 11.2 | 252 | no | no | no |
| wikitext | 92000 | cap1024_plain | 84208 | 909.7 | 121 | no | yes | yes |
| wikitext | 92000 | cap1024_strict | 84213 | 17.4 | 1024 | yes | yes | no |
| wikitext | 92000 | cap4096_plain | 84208 | 17.4 | 223 | no | yes | yes |
| wikitext | 92000 | cap4096_strict | 84213 | 17.2 | 167 | no | yes | yes |
| code | 46000 | cap1024_plain | 68573 | 671.8 | 151 | no | no | no |
| code | 46000 | cap1024_strict | 68578 | 15.7 | 1024 | yes | no | no |
| code | 46000 | cap4096_plain | 68573 | 15.0 | 4096 | yes | no | no |
| code | 46000 | cap4096_strict | 68578 | 15.6 | 4096 | yes | no | no |
| code | 92000 | cap1024_plain | 102834 | 1263.6 | 1024 | yes | no | no |
| code | 92000 | cap1024_strict | 102839 | 20.1 | 1024 | yes | no | no |
| code | 92000 | cap4096_plain | 102834 | 20.0 | 4096 | yes | no | no |
| code | 92000 | cap4096_strict | 102839 | 20.0 | 4096 | yes | no | no |
