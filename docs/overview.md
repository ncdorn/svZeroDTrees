# svZeroDTrees Overview

svZeroDTrees builds and adapts structured microvascular trees for svZeroD models. It supports both impedance and RCR outlet boundary conditions, optional tuning to clinical targets, and a full pipeline that can include 3D coupling via SimVascular tools.

**Where It Fits**
- Input: an svZeroD config and clinical targets.
- Output: boundary conditions assigned to outlets, optional adaptation results, and postprocess figures.
- Optional: 3D coupling using `svpre`, `svsolver`, and `svpost`.

**Inputs You Need**
- `zerod_config.json`: svZeroD configuration.
- `clinical_targets.csv`: target flow/pressure metrics used for tuning and adaptation.
- `mesh-surfaces` directory: outlet surface data from SimVascular preprocessing.
- Optional `inflow.csv`: custom inflow waveform.

**Typical Usage Paths**

Minimal path (tune + construct):
```bash
svzerodtrees tune-bcs examples/tune_bcs_example.yml
svzerodtrees construct-trees examples/construct_tree/construct_trees_example.yml
```

Full pipeline:
```bash
svzerodtrees pipeline examples/pipeline_example.yml
```

Postprocess saved trees:
```bash
svzerodtrees postprocess examples/postprocess_example.yml
```

**Step-by-Step Sketch**
1. Provide `paths` to your svZeroD config, clinical targets, and mesh-surfaces.
2. Choose `bcs.type` as `impedance` or `rcr`.
3. Run `tune_bcs` to compute optimized parameters.
4. Run `construct_trees` to build BCs and write a config with BCs attached.
5. Run `adapt` to apply microvascular adaptation to preop/postop results.
6. Run `postprocess` to generate figures from saved tree pickles.

**Common Gotchas**
- The `mesh-surfaces` directory must come from SimVascular preprocessing and must exist.
- 3D coupling requires `svpre`, `svsolver`, and `svpost` to be available in PATH.
- Relative paths are resolved relative to `paths.root`.

**Short YAML Examples**
Minimal `tune_bcs`:
```yaml
version: 1
workflow: tune_bcs
paths:
  root: .
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
bcs:
  type: impedance
  compliance_model: constant
  is_pulmonary: true
```

Minimal `construct_trees`:
```yaml
version: 1
workflow: construct_trees
paths:
  root: .
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
bcs:
  type: impedance
  compliance_model: constant
  is_pulmonary: true
trees:
  d_min: 0.01
  use_mean: true
  specify_diameter: true
  optimized_params_csv: optimized_params.csv
```

**Where to Go Next**
- Full schema reference: `docs/interface.md`.
- Complete examples: `examples/*.yml` and `examples/construct_tree/README.md`.
