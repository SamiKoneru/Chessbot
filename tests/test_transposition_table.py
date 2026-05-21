import unittest

from bot.move import Move
from bot.transposition_table import BoundType, TranspositionEntry, TranspositionTable


class TranspositionTableTests(unittest.TestCase):
    def test_store_and_lookup_round_trip(self) -> None:
        table = TranspositionTable()
        entry = TranspositionEntry(
            zobrist_hash=123,
            depth=4,
            score=91,
            bound=BoundType.EXACT,
            best_move=Move("e2", "e4"),
        )

        table.store_entry(entry)
        self.assertEqual(table.lookup(123), entry)

    def test_shallower_entry_does_not_replace_deeper_entry(self) -> None:
        table = TranspositionTable()
        table.store(
            zobrist_hash=123,
            depth=5,
            score=42,
            bound=BoundType.EXACT,
            best_move=Move("e2", "e4"),
        )
        table.store(
            zobrist_hash=123,
            depth=3,
            score=17,
            bound=BoundType.LOWER,
            best_move=Move("d2", "d4"),
        )

        entry = table.lookup(123)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.depth, 5)
        self.assertEqual(entry.score, 42)
        self.assertEqual(entry.best_move, Move("e2", "e4"))

    def test_same_or_deeper_entry_replaces_existing_entry(self) -> None:
        table = TranspositionTable()
        table.store(
            zobrist_hash=123,
            depth=3,
            score=17,
            bound=BoundType.LOWER,
            best_move=Move("d2", "d4"),
        )
        table.store(
            zobrist_hash=123,
            depth=4,
            score=91,
            bound=BoundType.EXACT,
            best_move=Move("e2", "e4"),
        )

        entry = table.lookup(123)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.depth, 4)
        self.assertEqual(entry.score, 91)
        self.assertEqual(entry.bound, BoundType.EXACT)
        self.assertEqual(entry.best_move, Move("e2", "e4"))


if __name__ == "__main__":
    unittest.main()
