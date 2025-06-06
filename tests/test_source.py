#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from pathlib import Path
from unittest import TestCase

import fsspec
import pytest
import s3fs
import zarr

from xarray_eopf.source import normalize_source, normalize_source_path


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


class GetSourcePathsTest(TestCase):
    def test_from_path(self):
        # From str
        paths = normalize_source_path("test1.zarr")
        self.assertIsInstance(paths, tuple)
        self.assertEqual(("test1.zarr", ""), paths)

        # From str with sub-group
        paths = normalize_source_path("s3://eopf-samples/data/test2.zarr/r10m")
        self.assertIsInstance(paths, tuple)
        self.assertEqual(("s3://eopf-samples/data/test2.zarr", "r10m"), paths)

        # From pathlib.Path
        paths = normalize_source_path(Path("test2.zarr"))
        self.assertIsInstance(paths, tuple)
        self.assertEqual(("test2.zarr", ""), paths)

        # From pathlib.Path with subgroup
        paths = normalize_source_path(Path("test2.zarr/r10m"))
        self.assertIsInstance(paths, tuple)
        self.assertEqual(("test2.zarr", "r10m"), paths)

    def test_from_mappings(self):
        # From fsspec.FSMap
        paths = normalize_source_path(
            fsspec.filesystem("local").get_mapper("test3.zarr")
        )
        self.assertIsInstance(paths, tuple)
        root_path, group_path = paths
        self.assertEqual("test3.zarr", Path(root_path.root).name)
        self.assertEqual("", group_path)

        # From zarr.storage.DirectoryStore
        paths = normalize_source_path(zarr.storage.DirectoryStore("test4.zarr"))
        self.assertIsInstance(paths, tuple)
        root_path, group_path = paths
        self.assertEqual("test4.zarr", Path(root_path.path).name)
        self.assertEqual("", group_path)

    def test_fail(self):
        # From dict
        self.assertEqual(
            ({"path": "test5.zarr"}, None),
            normalize_source_path({"path": "test5.zarr"}),
        )
