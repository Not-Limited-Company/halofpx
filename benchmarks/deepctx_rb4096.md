# Deep-context probe - rb4096

- Needle at 50% of the material; conditions vary the tail only
- Notes: realistic content (wikipedia prose + this repo's sources); needle at 50%; reasoning budget 4096

| content | target tokens | condition | prompt tokens | prefill s | generated | hit cap | retrieved | correct answer |
|---|---:|---|---:|---:|---:|:--:|:--:|:--:|
| wikitext | 46000 | cap1024_plain | 42309 | 326.8 | 479 | no | no | no |
| wikitext | 46000 | cap1024_strict | 42314 | 11.3 | 591 | no | no | no |
| wikitext | 46000 | cap4096_plain | 42309 | 10.9 | 4096 | yes | no | no |
| wikitext | 46000 | cap4096_strict | 42314 | 11.2 | 4096 | yes | no | no |
| wikitext | 92000 | cap1024_plain | 84209 | 908.2 | 111 | no | yes | yes |
| wikitext | 92000 | cap1024_strict | 84214 | 17.4 | 284 | no | no | no |
| wikitext | 92000 | cap4096_plain | 84209 | 17.3 | 130 | no | yes | yes |
| wikitext | 92000 | cap4096_strict | 84214 | 17.2 | 4096 | yes | no | no |
| code | 46000 | cap1024_plain | 62330 | 582.6 | 1024 | yes | no | no |
| code | 46000 | cap1024_strict | 62335 | 14.5 | 1024 | yes | no | no |
| code | 46000 | cap4096_plain | 62330 | 13.9 | 4096 | yes | no | no |
| code | 46000 | cap4096_strict | 62335 | 14.3 | 4096 | yes | no | no |
| code | 92000 | cap1024_plain | 124608 | 1734.8 | 1024 | yes | no | no |
| code | 92000 | cap1024_strict | 124613 | 23.2 | 1024 | yes | no | no |
| code | 92000 | cap4096_plain | 124608 | 22.9 | 4096 | yes | no | no |
| code | 92000 | cap4096_strict | 124613 | 23.0 | 4096 | yes | no | no |
