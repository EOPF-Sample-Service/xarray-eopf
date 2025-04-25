#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import fsspec
import s3fs

from .constants import DEFAULT_ENDPOINT_URL


def normalize_source(source: Any, storage_options: Mapping[str, Any] | None) -> Any:
    if isinstance(source, (str, Path)):
        protocol, root = fsspec.core.split_protocol(source)
        if protocol == "s3":
            return _get_s3_store(root, storage_options)
    else:
        if storage_options is not None:
            raise ValueError("storage_options argument applies only to paths or URLs")
    return source


def get_source_paths(source: Any) -> tuple[str, str] | tuple[None, None]:
    """
    Derive a Zarr root path (or URL) and subgroup path
    from given `source` object.

    Args:
        source: The source object.

    Returns:
        A tuple comprising a Zarr root path and subgroup path,
        or the tuple `(None, None)` if the paths cannot be derived.
        The subgroup path will be the empty string if it is
        not part of the path derived from `source`.
    """
    path: str | None = None
    if isinstance(source, (str, Path)):
        path = source
    elif hasattr(source, "path"):
        path = source.path
    elif hasattr(source, "root"):
        path = source.root
    if isinstance(path, (str, Path)):
        path_parts = str(path).split(".zarr/", maxsplit=1)
        if len(path_parts) == 2:
            return path_parts[0] + ".zarr", path_parts[1]
        return path_parts[0], ""
    return None, None


def _get_s3_store(root: str, storage_options: Mapping[str, Any] | None) -> fsspec.FSMap:
    # CEPH uses a non-standard colon to separate tenant name from
    # the bucket name. We need to convince boto3 to work with that.
    storage_options = storage_options or {}
    is_ceph_fs = ":" in root
    if (
        "anon" not in storage_options
        and "client" not in storage_options
        and "secret" not in storage_options
    ):
        storage_options["anon"] = True
    if (
        is_ceph_fs
        and "endpoint_url" not in storage_options
        and "endpoint_url" not in storage_options.get("client_kwargs", {})
    ):
        storage_options["endpoint_url"] = DEFAULT_ENDPOINT_URL

    s3_fs = s3fs.S3FileSystem(**storage_options)
    if is_ceph_fs:
        # The following is a hack to force boto3 to deal with colons
        # in bucket names.
        # First unregister handler to make boto3 work with CEPH
        # noinspection PyProtectedMember
        handlers = s3_fs.s3.meta.events._emitter._handlers
        handlers_to_unregister = handlers.prefix_search("before-parameter-build.s3")
        if len(handlers_to_unregister):
            # The first handler should be the function 'validate_bucket_name()'
            handler_to_unregister = handlers_to_unregister[0]
            # noinspection PyProtectedMember
            s3_fs.s3.meta.events._emitter.unregister(
                "before-parameter-build.s3", handler_to_unregister
            )

    return s3_fs.get_mapper(root=root, create=False, check=False)
