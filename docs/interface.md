# svzerodtrees YAML Interface (v1)

This document defines the YAML schema used by the CLI and Python API. The schema is strict: unknown keys raise errors.

**Workflows**
- `pipeline`: end-to-end run using `Simulation.run_pipeline`.
- `tune_bcs`: tune boundary conditions only.
- `construct_trees`: assign impedance or RCR BCs.
- `adapt`: run microvascular adaptation (impedance BCs).
- `postprocess`: generate figures from saved tree pickles.

**Workflow Requirements**

| Workflow | Required Sections | Optional Sections |
| --- | --- | --- |
| `pipeline` | `version`, `workflow`, `paths` | `bcs`, `adaptation`, `pipeline`, `threed` |
| `tune_bcs` | `version`, `workflow`, `paths`, `bcs` | `threed` |
| `construct_trees` | `version`, `workflow`, `paths`, `bcs`, `trees` | `threed` |
| `adapt` | `version`, `workflow`, `paths` | `bcs`, `adaptation`, `threed` |
| `postprocess` | `version`, `workflow`, `paths`, `postprocess` | none |

**Path Resolution**
- `paths.root` is resolved to an absolute path.
- All other relative paths are resolved relative to `paths.root`.

**Top-Level Keys**
- `version`: must be `1`.
- `workflow`: one of `pipeline|tune_bcs|construct_trees|adapt|postprocess`.
- `paths`: required for all workflows.
- `bcs`, `trees`, `adaptation`, `pipeline`, `threed`, `postprocess`: required only for certain workflows.

**Paths**
```yaml
paths:
  root: .
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-complete/mesh-surfaces
  preop_dir: preop
  postop_dir: postop
  adapted_dir: adapted
  inflow: inflow.csv
  optimized_params: optimized_params.csv
  output_config: svzerod_config_with_bcs.json
```

**BCs**
```yaml
bcs:
  type: impedance  # impedance | rcr
  compliance_model: constant
  is_pulmonary: true
  tune_space:
    free:
      - name: lpa.alpha
        init: 0.9
        lb: 0.7
        ub: 0.99
        to_native: identity
        from_native: identity
    fixed:
      - name: d_min
        value: 0.01
    tied: []
  rcr_params: [R_LPA, C_LPA, R_RPA, C_RPA]
```

Supported transforms for `tune_space`:
- `to_native`: `identity|positive|unit_interval`
- `from_native`: `identity|log`

**Trees**
```yaml
trees:
  d_min: 0.01
  use_mean: true
  specify_diameter: true
  optimized_params_csv: optimized_params.csv
  lpa:
    lrr: 10.0
    diameter: 0.3
    d_min: 0.01
    alpha: 0.9
    beta: 0.6
    inductance: 0.0
    compliance:
      model: constant  # constant | olufsen
      params:
        value: 66000.0
  rpa:
    lrr: 10.0
    diameter: 0.3
    d_min: 0.01
    alpha: 0.9
    beta: 0.6
    inductance: 0.0
    compliance:
      model: constant
      params:
        value: 66000.0
```
If `optimized_params_csv` is present, `lpa` and `rpa` blocks are optional.

**Adaptation**
```yaml
adaptation:
  method: cwss
  location: uniform
  iterations: 10
```

**Pipeline**
```yaml
pipeline:
  run_steady: true
  optimize_bcs: true
  run_threed: true
  adapt: true
```

**3D**
```yaml
threed:
  mesh_scale_factor: 1.0
  convert_to_cm: false
  wall_model: rigid  # rigid | deformable
  elasticity_modulus: 5062674.563165
  poisson_ratio: 0.5
  shell_thickness: 0.12
  prestress_file: auto  # auto | from_steady_mean | path/to/prestress_result.vtu
  prestress_file_path: path/to/prestress_result.vtu
  solver_paths:
    svpre: svpre
    svsolver: svsolver
    svpost: postsolver
```

For deformable wall runs, setting `prestress_file: auto` creates and runs a `prestress/` simulation using mean wall traction from `steady/mean` results, then uses that resulting VTU as `Prestress_file_path`.

**Postprocess**
```yaml
postprocess:
  figures:
    - kind: generation_metrics
      input: path/to/tree.pkl
      output: figures/generation_metrics.png
      options:
        time_window: [0.0, 1.0]
        exclude_collapsed: true
```

Supported `kind` values:
- `generation_metrics`
- `generation_waveforms`
- `visualize_hemodynamics`

**CLI Usage**
```bash
svzerodtrees pipeline config.yml
svzerodtrees tune-bcs config.yml
svzerodtrees construct-trees config.yml
svzerodtrees adapt config.yml
svzerodtrees postprocess config.yml
svzerodtrees schema
```
