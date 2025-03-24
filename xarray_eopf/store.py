#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Mapping
from typing import Any

import fsspec
import s3fs

from xarray_eopf.constants import DEFAULT_ENDPOINT_URL


def open_store(
    filename_or_obj: str,
    protocol: str | None,
    storage_options: Mapping[str, Any] | None,
) -> Any:
    if isinstance(filename_or_obj, str):
        return _open_fs_store(filename_or_obj, protocol, storage_options)
    else:
        if protocol is not None:
            raise ValueError("the protocol argument applies only to paths or URLs")
        if storage_options is not None:
            raise ValueError(
                "the storage_options argument applies only to paths or URLs"
            )
        return filename_or_obj


def _open_fs_store(
    path_or_url: str, protocol: str | None, storage_options: Mapping[str, Any] | None
) -> fsspec.FSMap:
    _protocol, root = fsspec.core.split_protocol(path_or_url)
    protocol = protocol or _protocol or "file"
    # CEPH uses a non-standard colon to separate tenant name from
    # the bucket name. We need to convince boto3 to work with that.
    storage_options = storage_options or {}
    is_ceph_fs = False
    if protocol == "s3":
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

    fs = fsspec.filesystem(protocol, **storage_options)
    if is_ceph_fs and isinstance(fs, s3fs.S3FileSystem):
        s3_fs: s3fs.S3FileSystem = fs
        # unregister handler to make boto3 work with CEPH
        # noinspection PyProtectedMember
        handlers = s3_fs.s3.meta.events._emitter._handlers
        handlers_to_unregister = handlers.prefix_search("before-parameter-build.s3")
        if len(handlers_to_unregister):
            handler_to_unregister = handlers_to_unregister[0]
            # noinspection PyProtectedMember
            s3_fs.s3.meta.events._emitter.unregister(
                "before-parameter-build.s3", handler_to_unregister
            )

    return fs.get_mapper(root=root, create=False, check=False)
