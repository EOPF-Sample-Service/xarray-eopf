#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
from pathlib import Path

from collections.abc import Mapping
from typing import Any

import fsspec
import s3fs

from xarray_eopf.constants import DEFAULT_ENDPOINT_URL


def normalize_source(source: Any, storage_options: Mapping[str, Any] | None) -> Any:
    if isinstance(source, (str, Path)):
        protocol, root = fsspec.core.split_protocol(source)
        if protocol == "s3":
            return _get_s3_store(root, storage_options)
    else:
        if storage_options is not None:
            raise ValueError("storage_options argument applies only to paths or URLs")
    return source


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
