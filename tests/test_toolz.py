import itertools
import operator
import random
import statistics
from typing import Tuple

import pytest
from toolz import (
    curried,
    functoolz,
    itertoolz,
)

import onekit.pythonkit as pk


class TestToolzAPI:
    def test_concat(self):
        """Concatenate zero or more iterables."""
        actual = tuple(itertoolz.concat([[], [1], [2, 3], [4, 5, 6]]))
        expected = (1, 2, 3, 4, 5, 6)
        assert actual == expected

    def test_concatv(self):
        """Variadic version of concat."""
        actual = tuple(itertoolz.concatv([], [1], [2, 3], [4, 5, 6]))
        expected = (1, 2, 3, 4, 5, 6)
        assert actual == expected

    def test_first(self, n10: Tuple[int]):
        """Take the first element."""
        actual = itertoolz.first(n10)
        expected = 1
        assert actual == expected

    def test_last(self, n10: Tuple[int]):
        """Take the last element."""
        actual = itertoolz.last(n10)
        expected = 10
        assert actual == expected

    def test_rest(self, n10: Tuple[int]):
        """Drop the first element and return the remaining elements."""
        actual = tuple(itertoolz.rest(n10))
        expected = (2, 3, 4, 5, 6, 7, 8, 9, 10)
        assert actual == expected

    def test_second(self, n10: Tuple[int]):
        """Take the second element."""
        actual = itertoolz.second(n10)
        expected = 2
        assert actual == expected

    def test_accumulate(self):
        """Accumulate elements in sequence."""
        cumsum = curried.accumulate(operator.add)
        cummul = curried.accumulate(operator.mul)

        actual = tuple(cumsum([1, 2, 3, 4, 5]))
        expected = (1, 3, 6, 10, 15)
        assert actual == expected

        actual = tuple(cummul([1, 2, 3, 4, 5]))
        expected = (1, 2, 6, 24, 120)
        assert actual == expected

        actual = functoolz.pipe(
            [1, 2, 3],
            curried.accumulate(operator.add, initial=-1),
            tuple,
        )
        expected = (-1, 0, 2, 5)
        assert actual == expected

    def test_cons(self):
        """Add element to the beginning of an in principle infinite sequence."""
        actual = tuple(curried.cons(10, [2, 3]))
        expected = (10, 2, 3)
        assert actual == expected

    def test_do(self):
        """Apply function f(x) = y, yet do not return y but input x."""
        actual = curried.do(lambda x: x**2, 4)
        expected = 4
        assert actual == expected

        log = []
        f = functoolz.compose(self.inc, curried.do(log.append))

        actual = f(1)
        expected = 2
        assert actual == expected
        assert log == [1]

        actual = f(11)
        expected = 12
        assert actual == expected
        assert log == [1, 11]

    def test_drop(self, n10: Tuple[int]):
        """Drop the first three elements and return the rest."""
        actual = tuple(curried.drop(3, n10))
        expected = (4, 5, 6, 7, 8, 9, 10)
        assert actual == expected

    def test_filter(self, n10: Tuple[int]):
        """Filter even numbers."""
        actual = tuple(filter(pk.iseven, n10))
        expected = (2, 4, 6, 8, 10)
        assert actual == expected

    def test_get(self, n10: Tuple[int]):
        """Get element in a sequence or dict."""
        actual = curried.get(1, n10)
        expected = 2
        assert actual == expected

        actual = curried.get([1, 5], n10)
        expected = (2, 6)
        assert actual == expected

        actual = curried.get(100, n10, default=-1)
        expected = -1
        assert actual == expected

        phone_book = {"Alice": "555-1234", "Bob": "555-5678", "Charlie": "555-9999"}
        actual = curried.get("Alice", phone_book)
        expected = "555-1234"
        assert actual == expected

        actual = curried.get("David", phone_book, None)
        expected = None
        assert actual == expected

        actual = curried.get(["Alice", "Bob"], phone_book)
        expected = ("555-1234", "555-5678")
        assert actual == expected

    def test_groupby(self, n10: Tuple[int]):
        """Group an iterable by a key function."""
        actual = curried.groupby(pk.iseven, n10)
        expected = {False: [1, 3, 5, 7, 9], True: [2, 4, 6, 8, 10]}
        assert actual == expected

    def test_interpose(self):
        """Insert element between each pair of elements in sequence."""
        actual = tuple(curried.interpose("a", [1, 2, 3, 4, 5]))
        expected = (1, "a", 2, "a", 3, "a", 4, "a", 5)
        assert actual == expected

    def test_iterate(self):
        """Repeatedly apply a function onto an original input."""
        counter = curried.iterate(self.inc, 0)
        assert next(counter) == 0
        assert next(counter) == 1
        assert next(counter) == 2

    def test_mapcat(self):
        """Apply a function to each sequence in seqs and concatenate results."""
        actual = tuple(
            curried.mapcat(curried.map(str.upper), [["a", "b"], ["c", "d", "e"]])
        )
        expected = ("A", "B", "C", "D", "E")
        assert actual == expected

    def test_nth(self, n10: Tuple[int]):
        """Take the nth element."""
        actual = curried.nth(4, n10)
        expected = 5
        assert actual == expected

    def test_partition(self):
        """Partition sequence into tuples of length n."""
        actual = tuple(curried.partition(2, [1, 2, 3, 4, 5], pad=None))
        expected = (
            (1, 2),
            (3, 4),
            (5, None),
        )
        assert actual == expected

    def test_partition_all(self):
        """Partition sequence into tuples of length n."""
        actual = tuple(curried.partition_all(2, [1, 2, 3, 4, 5]))
        expected = (
            (1, 2),
            (3, 4),
            (5,),
        )
        assert actual == expected

    def test_peekn(self):
        """Retrieve the next n elements of a sequence.

        Returns a tuple of the first n elements and an iterable equivalent
        to the original, still having the elements retrieved.
        """
        seq = (0, 1, 2, 3, 4)
        first_two, seq_with_all_original_values = curried.peekn(2, seq)
        assert isinstance(first_two, tuple) and first_two == seq[:2]
        assert tuple(seq_with_all_original_values) == seq

    def test_pluck(self):
        """Plucks an element or several elements from each item in a sequence."""
        data = [{"id": 1, "name": "Cheese"}, {"id": 2, "name": "Pies"}]
        actual = tuple(curried.pluck("name", data))
        expected = ("Cheese", "Pies")
        assert actual == expected

        actual = tuple(curried.pluck([0, 1], [[1, 2, 3], [4, 5, 7]]))
        expected = ((1, 2), (4, 5))
        assert actual == expected

    def test_random_sample(self, n10: Tuple[int]):
        """Return elements from a sequence with probability of prob.

        It considers each item independently and without replacement.
        Returns a lazy iterator of random items from seq.
        """
        rng = random.Random(2016)
        seq = [i - 1 for i in n10]
        actual = tuple(curried.random_sample(0.1, seq, random_state=rng))
        expected = (7, 9)
        assert actual == expected

        actual = tuple(curried.random_sample(0.1, seq, random_state=rng))
        expected = (9,)
        assert actual == expected

    def test_reduceby(self):
        """Perform a simultaneous groupby and reduction."""
        actual = curried.reduceby(pk.iseven, operator.add, [1, 2, 3, 4, 5])
        expected = {False: 9, True: 6}
        assert actual == expected

    def test_remove(self):
        """Remove those elements of sequence for which predicate(item) is True."""
        actual = tuple(curried.remove(pk.iseven, [1, 2, 3, 4]))
        expected = (1, 3)
        assert actual == expected

    def test_sliding_window(self):
        """A sequence of overlapping subsequences.

        This function creates a sliding window suitable for transformations like
        sliding means / smoothing.
        """
        actual = tuple(curried.sliding_window(2, [1, 2, 3, 4]))
        expected = ((1, 2), (2, 3), (3, 4))
        assert actual == expected

        actual = functoolz.pipe(
            [1, 2, 3, 4],
            curried.sliding_window(2),
            curried.map(statistics.mean),
            tuple,
        )
        expected = (1.5, 2.5, 3.5)
        assert actual == expected

    def test_tail(self, n10: Tuple[int]):
        """Take the last nth elements."""
        actual = tuple(curried.tail(3, n10))
        expected = (8, 9, 10)
        assert actual == expected

    def test_take(self):
        """Take the first nth elements."""
        actual = tuple(curried.take(3, itertools.count(1)))
        expected = (1, 2, 3)
        assert actual == expected

    def test_take_nth(self, n10: Tuple[int]):
        """Take every nth element."""
        actual = tuple(curried.take_nth(2, n10))
        expected = (1, 3, 5, 7, 9)
        assert actual == expected

    def test_topk(self):
        """Find the k-largest elements of a sequence.

        Operates lazily in `n * log(k)` time.
        """
        actual = curried.topk(2, [1, 100, 10, 1000])
        expected = (1000, 100)
        assert actual == expected

        actual = curried.topk(2, ["Alice", "Bob", "Charlie", "Dan"], key=len)
        expected = ("Charlie", "Alice")
        assert actual == expected

    def test_unique(self):
        """Return only unique elements of a sequence."""
        assert set(curried.unique((1, 3, 2))) == {1, 2, 3}
        assert set(curried.unique((1, 3, 2, 1, 3))) == {1, 2, 3}

        actual = set(curried.unique(["cat", "mouse", "dog", "hen"], key=len))
        expected = {"cat", "mouse"}
        assert actual == expected

        actual = set(curried.unique(["dog", "mouse", "cat", "hen"], key=len))
        expected = {"dog", "mouse"}
        assert actual == expected

    def test_countby(self):
        """Count elements of an iterable by a key function."""
        actual = functoolz.pipe(
            itertools.count(1),
            curried.take(11),
            curried.countby(pk.iseven),
        )
        expected = {True: 5, False: 6}
        assert actual == expected

    def test_partitionby(self):
        """Partition a sequence according to a function."""
        actual = tuple(
            curried.partitionby(
                lambda x: x > 10,
                [1, 2, 1, 99, 88, 33, 3, 10, 99, -1, 5],
            )
        )
        expected = (
            (1, 2, 1),
            (99, 88, 33),
            (3, 10),
            (99,),
            (-1, 5),
        )
        assert actual == expected

    def test_assoc(self):
        """Update dictionary."""
        d = {"x": 1}
        actual = curried.assoc(d, "x", 2)
        expected = {"x": 2}
        assert actual == expected

        d = {"x": 1}
        actual = curried.assoc(d, "y", 3)
        expected = {"x": 1, "y": 3}
        assert actual == expected

    def test_assoc_in(self):
        """Return a new dict with new, potentially nested, key value pair."""
        purchase = {
            "name": "Alice",
            "order": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
            "credit card": "5555-1234-1234-1234",
        }
        actual = curried.assoc_in(purchase, ["order", "costs"], [0.25, 1.00])
        expected = {
            "name": "Alice",
            "order": {"items": ["Apple", "Orange"], "costs": [0.25, 1.00]},
            "credit card": "5555-1234-1234-1234",
        }
        assert actual == expected

    def test_dissoc(self):
        """Return a new dict with the given key(s) removed."""
        actual = curried.dissoc({"x": 1, "y": 2}, "y")
        expected = {"x": 1}
        assert actual == expected

        actual = curried.dissoc({"x": 1, "y": 2}, "y", "x")
        expected = {}
        assert actual == expected

        actual = curried.dissoc({"x": 1}, "y")  # Ignores missing keys
        expected = {"x": 1}
        assert actual == expected

    def test_get_in(self):
        """It is a generalization of ``operator.getitem`` for nested
        data structures such as dictionaries and lists.
        """
        transaction = {
            "name": "Alice",
            "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
            "credit card": "5555-1234-1234-1234",
        }
        actual = curried.get_in(["purchase", "items", 0], transaction)
        expected = "Apple"
        assert actual == expected

    def test_itemfilter(self):
        """Filter items in dictionary by item."""

        def isvalid(item):
            k, v = item
            return k % 2 == 0 and v < 4

        d = {1: 2, 2: 3, 3: 4, 4: 5}
        actual = curried.itemfilter(isvalid, d)
        expected = {2: 3}
        assert actual == expected

    def test_itemmap(self):
        """Apply function to items of dictionary."""
        account_ids = {"Alice": 10, "Bob": 20}
        actual = curried.itemmap(reversed, account_ids)
        expected = {10: "Alice", 20: "Bob"}
        assert actual == expected

    def test_keyfilter(self):
        """Filter items in dictionary by key."""
        d = {1: "a", 2: "b", 3: "c", 4: "d"}
        actual = curried.keyfilter(pk.iseven, d)
        expected = {2: "b", 4: "d"}
        assert actual == expected

    def test_keymap(self):
        """Apply function to keys of dictionary."""
        bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
        actual = curried.keymap(str.lower, bills)
        expected = {"alice": [20, 15, 30], "bob": [10, 35]}
        assert actual == expected

    def test_merge(self):
        """Merge an iterable of dictionaries.
        Later dictionaries have precedence.
        """
        actual = curried.merge({1: "one"}, {2: "two"})
        expected = {1: "one", 2: "two"}
        assert actual == expected

        actual = curried.merge({1: "one"}, {2: "two"}, {2: "three"})
        expected = {1: "one", 2: "three"}
        assert actual == expected

    def test_merge_with(self):
        """Merge dictionaries and apply function to combined values."""
        actual = curried.merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
        expected = {1: 11, 2: 22}
        assert actual == expected

    def test_update_in(self):
        """Update value in a (potentially) nested dictionary."""
        actual = curried.update_in({"a": 0}, ["a"], self.inc)
        expected = {"a": 1}
        assert actual == expected

    def test_valfilter(self):
        """Filter items in dictionary by value."""
        d = {"a": 2, "b": 3, "c": 4, "d": 5}
        actual = curried.valfilter(pk.iseven, d)
        expected = {"a": 2, "c": 4}
        assert actual == expected

    def test_valmap(self):
        """Apply function to values of dictionary."""
        bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
        actual = curried.valmap(sum, bills)
        expected = {"Alice": 65, "Bob": 45}
        assert actual == expected

    @staticmethod
    def inc(n: int) -> int:
        return n + 1

    @pytest.fixture(scope="class")
    def n10(self) -> Tuple[int]:
        return 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
