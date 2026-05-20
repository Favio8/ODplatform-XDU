# ADR-002 Data Pipeline

- Status: accepted
- Decision: build a dedicated `odp_platform.data_pipeline` subsystem that converts raw datasets into split YOLO assets and emits Ultralytics dataset YAML files
- Scope:
  - converter registry with format-based lazy loading
  - unified `ConvertOptions` input object
  - orchestrated flow: coverage check -> conversion -> split -> materialization -> yaml generation
  - supported source formats in D3: `pascal_voc`, `coco`, `yolo`
- Rationale:
  - keep CLI, service, converter, split, and yaml responsibilities separated
  - allow new dataset formats to be added without changing the CLI contract
  - fail fast on broken raw datasets before expensive conversion or training steps
- Consequences:
  - `odp-transform` becomes the formal data preparation command
  - generated dataset YAML files live in `apps/platform/configs/datasets/`
  - future data formats should register capabilities through the registry rather than branching inside the CLI
