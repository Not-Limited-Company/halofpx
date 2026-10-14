# Deep-context probe - no_reasoning

- Needle at 50% of the material; conditions vary the tail only
- Notes: reasoning off; realistic content; needle at 50%

| content | target tokens | condition | prompt tokens | prefill s | generated | hit cap | retrieved | correct answer |
|---|---:|---|---:|---:|---:|:--:|:--:|:--:|
| wikitext | 46000 | cap1024_plain | 42309 | 326.5 | 11 | no | no | no |
| wikitext | 46000 | cap1024_strict | 42314 | 11.4 | 1024 | yes | no | no |
| wikitext | 46000 | cap4096_plain | 42309 | 11.0 | 11 | no | no | no |
| wikitext | 46000 | cap4096_strict | 42314 | 11.2 | 4096 | yes | no | no |
| wikitext | 92000 | cap1024_plain | 84209 | 908.9 | 10 | no | no | no |
| wikitext | 92000 | cap1024_strict | 84214 | 17.5 | 5 | no | no | no |
| wikitext | 92000 | cap4096_plain | 84209 | 17.4 | 10 | no | no | no |
| wikitext | 92000 | cap4096_strict | 84214 | 17.2 | 10 | no | no | no |
| code | 46000 | cap1024_plain | 62330 | 582.9 | 16 | no | no | no |
| code | 46000 | cap1024_strict | 62335 | 14.5 | 1024 | yes | no | no |
| code | 46000 | cap4096_plain | 62330 | 13.8 | 16 | no | no | no |
| code | 46000 | cap4096_strict | 62335 | 14.2 | 4096 | yes | no | no |
| code | 92000 | cap1024_plain | 124608 | 1738.0 | 1024 | yes | no | no |
| code | 92000 | cap1024_strict | 124613 | 23.2 | 1024 | yes | no | no |
| code | 92000 | cap4096_plain | 124608 | 22.8 | 4096 | yes | no | no |
| code | 92000 | cap4096_strict | 124613 | 23.0 | 4096 | yes | no | no |
