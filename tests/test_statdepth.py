#!/usr/bin/env python

"""Tests for `statdepth` package."""

import unittest
import pandas as pd 
import numpy as np

from statdepth import *
from statdepth.testing import *
from statdepth.homogeneity import * 

class TestStatdepth(unittest.TestCase):
    """Tests for `statdepth` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_functional(self):
        """Test something."""
        df = generate_noisy_univariate()
        bd = FunctionalDepth([df], containment='r2')

        self.assertIsInstance(bd, pd.Series)
        self.assertIsInstance(bd.ordered(), pd.Series)
        self.assertIsInstance(bd.median(), pd.Series)
        self.assertIsInstance(bd.deepest(n=2), pd.Series)
        self.assertIsInstance(bd.outlying(n=2), pd.Series)

        self.assertIsInstance(FunctionalDepth([df], K=5, containment='r2'), pd.Series)

    def test_pointcloud_l1(self):
        df = generate_noisy_pointcloud(n=10, d=2)
        bd = PointcloudDepth(df, containment='l1')
        self.assertIsInstance(bd, pd.Series)
        self.assertIsInstance(bd.ordered(), pd.Series)
        self.assertIsInstance(bd.median(), pd.Series)
        self.assertIsInstance(bd.deepest(n=2), pd.Series)
        self.assertIsInstance(bd.outlying(n=2), pd.Series)


        self.assertIsInstance(PointcloudDepth(df, K=2, containment='l1'), pd.Series)
    def test_pointcloud_simplex(self):
        df = generate_noisy_pointcloud(n=10, d=2)
        bd = PointcloudDepth(df, containment='simplex')
        self.assertIsInstance(bd, pd.Series)
        self.assertIsInstance(bd.ordered(), pd.Series)
        self.assertIsInstance(bd.median(), pd.Series)
        self.assertIsInstance(bd.deepest(n=2), pd.Series)
        self.assertIsInstance(bd.outlying(n=2), pd.Series)

        self.assertIsInstance(PointcloudDepth(df, K=2, containment='simplex'), pd.Series)

    def test_pointcloud_oja(self):

        df = generate_noisy_pointcloud(n=20, d=2)
        bd = PointcloudDepth(df, containment='oja')
        self.assertIsInstance(bd, pd.Series)
        self.assertIsInstance(bd.ordered(), pd.Series)
        self.assertIsInstance(bd.median(), pd.Series)
        self.assertIsInstance(bd.deepest(n=2), pd.Series)
        self.assertIsInstance(bd.outlying(n=2), pd.Series)

        self.assertIsInstance(PointcloudDepth(df, K=2, containment='oja'), pd.Series)

    def test_multivariate(self):
        data = generate_noisy_multivariate()
        bd = FunctionalDepth(data, containment='simplex')
        self.assertIsInstance(bd, pd.Series)
        self.assertIsInstance(bd.ordered(), pd.Series)

if __name__ == "__main__":
    unittest.main()