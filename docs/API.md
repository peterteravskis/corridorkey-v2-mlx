# Python API

[Home](../README.md) · [CLI](CLI.md) · [Profiles](MODELS.md)

The model loader accepts a directory of exported component profiles. It does
not execute Python from a model repository and does not download files itself.
Only use model assets from a source you trust.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from corridorkey_mlx.mlx_backend import MLXModel
from corridorkey_mlx.pipeline import Source, run

model = MLXModel(
    root="/path/to/CorridorKey-v2-MLX",
    height=1024,
    width=1536,
    compile_graph=True,
)

destination = Path("outputs/example")
destination.mkdir(parents=True, exist_ok=True)
source = Source(Path("plate.png"))
assert source.padded == (1024, 1536)
with TemporaryDirectory(prefix=".corridorkey-", dir=destination) as scratch:
    run(model, source, destination, scratch,
        outputs=("alpha", "fg", "raw_fg", "checker", "cutout"),
        png=True, screen="green", despill=1.0)
```

Reuse the model for further sources of the same padded shape. The `run` function
starts a new recurrent state for each source. `cache_mib=256` controls MLX's
allocator cache, not the size of live weights/activations. Concurrent calls on
the same model are not validated.

The lower-level `MLXGraph` interface loads `graph.json` and
`weights.safetensors`, accepts the ordered inputs described by the graph, and
returns MLX arrays. `MLXModel` provides CPU-tensor bridges for foundation,
stage-one, and stage-two integration. Neural operations execute on MLX.
The model/profile assets retain their upstream licenses regardless of which
API is used to load them.
