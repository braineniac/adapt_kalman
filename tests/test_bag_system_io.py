#!/usr/bin/env python

import sys
import unittest
import rosunit

from adapt_kalman.bag_system_io import BagSystemIO


class TestBagSystemIO(unittest.TestCase):
    def test_init(self):
        self.assertTrue(BagSystemIO())

    def test_input(self):
        bag_system_io = BagSystemIO()
        input = [(0.1, (1, 2, 2, 2, 2, 2)), (0.1, (1, 2, 2, 2, 2, 2)), (0.1, (1, 2, 2, 2, 2, 2))]
        output = [(0.1, (1, 2)), (0.1, (1, 2)), (0.1, (1, 2))]
        self.assertEqual(bag_system_io.get_input(input), output)

    def test_output(self):
        bag_system_io = BagSystemIO()
        input = [(0.1, (1, 2, 2, 2, 2, 2)), (0.1, (1, 2, 2, 2, 2, 2)), (0.1, (1, 2, 2, 2, 2, 2))]
        output = [(0.1, (1, 2)), (0.1, (1, 2)), (0.1, (1, 2))]
        self.assertEqual(bag_system_io.get_output(input), output)

    def test_states(self):
        bag_system_io = BagSystemIO()
        input = [(0.1, (1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)), (0.1, (1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1))]
        output = [(0.1, (1,1,1,1,1,1)),(0.1, (1,1,1,1,1,1))]
        self.assertEqual(bag_system_io.get_states(input),output)

if __name__ == '__main__':
    rosunit.unitrun("adapt_kalman", 'test_bag_system_io', TestBagSystemIO)
