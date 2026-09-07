---
dataset_info:
  features:
  - name: task_id
    dtype: string
  - name: prompt
    dtype: string
  - name: test
    dtype: string
  - name: original_test
    dtype: string
  - name: impossible_type
    dtype: string
  - name: entry_point
    dtype: string
  splits:
  - name: conflicting
    num_bytes: 676794
    num_examples: 103
  - name: oneoff
    num_bytes: 674810
    num_examples: 103
  - name: original
    num_bytes: 645775
    num_examples: 103
  download_size: 875550
  dataset_size: 1997379
configs:
- config_name: default
  data_files:
  - split: conflicting
    path: data/conflicting-*
  - split: oneoff
    path: data/oneoff-*
  - split: original
    path: data/original-*
---
