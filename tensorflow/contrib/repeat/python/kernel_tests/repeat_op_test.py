# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Repeat."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.repeat.python.ops import repeat
from tensorflow.python.platform import test

class RepeatTest(test.TestCase):
  
  def _testRepeat(self, input, repeats, axis, use_gpu=False, expected_err=None):
    if expected_err is None:
      np_repeat = np.repeat(input, repeats, axis)
      tf_repeat_tensor = repeat(input, repeats, axis)
      with self.test_session(use_gpu=use_gpu):
        tf_repeat = tf_repeat_tensor.eval()
      self.assertAllClose(np_repeat, tf_repeat)
      self.assertShapeEqual(np_repeat, tf_repeat_tensor)
    else:
      with self.test_session(use_gpu=use_gpu):
        with self.assertRaisesOpError(expected_err):
          repeat(input, repeats, axis).eval()
    
  def _testScalar(self, dtype):
    input = np.asarray(100 * np.random.randn(200), dtype=dtype)
    repeats = 2
    axis = 0
    self._testRepeat(input, repeats, axis)
    
    input = np.asarray(100 * np.random.randn(3, 2, 4, 5, 6), dtype=dtype)
    repeats = 3
    axis = 1
    self._testRepeat(input, repeats, axis)
    
  def _testVector(self, dtype):
    input = np.asarray(100 * np.random.randn(200), dtype=dtype)
    repeats = np.asarray(10 * np.random.randn(200), dtype=np.int32) % 5
    axis = 0
    self._testRepeat(input, repeats, axis)
    
    input = np.asarray(100 * np.random.randn(3, 2, 4, 5, 6), dtype=dtype)
    repeats = np.asarray(10 * np.random.randn(4), dtype=np.int32) % 5
    axis = 2
    self._testRepeat(input, repeats, axis)
    
  def testFloat(self):
    self._testScalar(np.float32)
    self._testVector(np.float32)

  def testDouble(self):
    self._testScalar(np.float64)
    self._testVector(np.float64)

  def testInt32(self):
    self._testScalar(np.int32)
    self._testVector(np.int32)

  def testInt64(self):
    self._testScalar(np.int64)
    self._testVector(np.int64)
  
if __name__ == "__main__":
  test.main()
