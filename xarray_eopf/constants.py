#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from typing import Final, Literal, TypeAlias

OP_MODE_ANALYSIS: Final = "analysis"
OP_MODE_NATIVE: Final = "native"
OP_MODES: Final = OP_MODE_ANALYSIS, OP_MODE_NATIVE

OpMode: TypeAlias = Literal["analysis", "native"]

# Keywords arguments passed to dataset.merge(other) when flattening
# data trees.
DS_MERGE_KWARGS: Final = dict(
    # skip comparing and pick variable from `dataset`
    compat="override",
    # use indexes from `dataset` that are the same size
    # as those of `other` in that dimension
    join="override",
    # skip comparing and copy attrs from `dataset` to
    # the result.
    combine_attrs="override",
)

DEFAULT_ENDPOINT_URL = "https://objectstore.eodc.eu:2222"
