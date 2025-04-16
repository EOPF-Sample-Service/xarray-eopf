#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
from pathlib import Path
from unittest import TestCase

import fsspec
import pytest
import s3fs

from xarray_eopf.source import normalize_source


class NormalizeSourceTest(TestCase):
    def test_s3_url(self):
        url = "s3://no-bucket/test.zarr"
        store = normalize_source(url, None)
        self.assertIsInstance(store, fsspec.FSMap)
        self.assertIsInstance(store.fs, s3fs.S3FileSystem)
        self.assertEqual("no-bucket/test.zarr", store.root)

    def test_ceph_s3_url(self):
        ceph_url = "s3://no-bucket:e6f4/test.zarr"
        store = normalize_source(ceph_url, None)
        self.assertIsInstance(store, fsspec.FSMap)
        self.assertIsInstance(store.fs, s3fs.S3FileSystem)
        self.assertEqual("no-bucket:e6f4/test.zarr", store.root)

    def test_https_url(self):
        path = "https://unknown.object.storage.com/no-bucket/test.zarr"
        source = normalize_source(path, None)
        self.assertEqual(path, source)

    def test_other(self):
        mapping = {}
        self.assertIs(mapping, normalize_source(mapping, None))

        path = Path("data/test.zarr")
        self.assertIs(path, normalize_source(path, None))

    # noinspection PyMethodMayBeStatic
    def test_fail(self):
        with pytest.raises(
            ValueError, match="storage_options argument applies only to paths or URLs"
        ):
            normalize_source({}, {})
