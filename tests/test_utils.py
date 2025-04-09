#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

from xarray_eopf.utils import NameFilter


class NameFilterTest(TestCase):
    def test_accept_name(self):
        f = NameFilter(includes=("ernie", "bert"))
        self.assertTrue(f.accept("ernie"))
        self.assertTrue(f.accept("bert"))
        self.assertFalse(f.accept("bibo"))

        f = NameFilter(includes=("ernie", "bert"), excludes="bert")
        self.assertTrue(f.accept("ernie"))
        self.assertFalse(f.accept("bert"))
        self.assertFalse(f.accept("bibo"))

    def test_accept_prefix(self):
        f = NameFilter(includes=("er", "be"))
        self.assertTrue(f.accept("ernie"))
        self.assertTrue(f.accept("bert"))
        self.assertFalse(f.accept("bibo"))

        f = NameFilter(includes=("er", "be"), excludes="be")
        self.assertTrue(f.accept("ernie"))
        self.assertFalse(f.accept("bert"))
        self.assertFalse(f.accept("bibo"))

    def test_accept_pattern(self):
        f = NameFilter(includes="e.*e")
        self.assertTrue(f.accept("ernie"))
        self.assertFalse(f.accept("erno"))
        self.assertFalse(f.accept("bert"))
        self.assertFalse(f.accept("bibo"))

    def test_filter(self):
        f = NameFilter(includes="e.*e")
        self.assertEqual(
            ["ernie", "emmie"], list(f.filter(["bibo", "ernie", "bert", "emmie"]))
        )
