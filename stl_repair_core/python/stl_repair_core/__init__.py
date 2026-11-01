# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""stl_repair_core: conservative STL intake, diagnosis, repair and verification
for structural-mechanics preprocessing.  Extracted from Sutura 0.9.8
(https://github.com/Krateian/Sutura, commit 158d9ab)."""
__version__ = "0.1.0"
SUTURA_SOURCE = "https://github.com/Krateian/Sutura/commit/158d9ab688fd43098cdd7fe8ad58f8f0049c34f9"

from .repair import repair_stl, validate_stl, DEFAULTS  # noqa: E402,F401
from .stlio import StlError, read_stl, write_stl  # noqa: E402,F401
