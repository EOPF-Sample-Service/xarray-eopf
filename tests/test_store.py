#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase
import fsspec
import s3fs
from xarray_eopf.store import open_store


class OpenStoreTest(TestCase):
    def test_s3_url(self):
        store = open_store("s3://no-bucket/test.zarr", None, None)
        self.assertIsInstance(store, fsspec.FSMap)
        self.assertIsInstance(store.fs, s3fs.S3FileSystem)

    def test_https_url(self):
        store = open_store(
            "https://unknown.object.storage.com/no-bucket/test.zarr", None, None
        )
        self.assertIsInstance(store, fsspec.FSMap)
        self.assertIsInstance(store.fs, fsspec.get_filesystem_class("http"))

    def test_other(self):
        filename_or_obj = {}
        # noinspection PyTypeChecker
        store = open_store(filename_or_obj, None, None)
        self.assertIs(store, filename_or_obj)
