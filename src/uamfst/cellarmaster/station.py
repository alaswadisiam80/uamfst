from __future__ import annotations

from collections.abc import Sequence

from absl import flags

from uamfst.cuverie import assemblage
from uamfst.cuverie.assemblage import Assembly, assemble
from uamfst.cuverie.manifold import clear, ingest

_ = assemblage

_GIN_FILE = flags.DEFINE_multi_string("gin_file", [], "Gin configuration files to parse in order.")
_GIN_BINDING = flags.DEFINE_multi_string("gin_binding", [], "Individual gin bindings to override.")


def build_assembly(files: Sequence[str], bindings: Sequence[str] | None = None) -> Assembly:
    clear()
    ingest(files, bindings or [])
    return assemble()


def configure() -> Assembly:
    return build_assembly(list(_GIN_FILE.value), list(_GIN_BINDING.value))
