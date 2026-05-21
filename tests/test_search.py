import unittest

from bot.board import Board
from bot.evaluation import CHECKMATE_SCORE
from bot.enums import PieceType
from bot.move import Move
from bot.search import alpha_beta_search, choose_move, iterative_deepening_search, ordered_moves
from bot.transposition_table import BoundType, TranspositionTable


class SearchTests(unittest.TestCase):
    def test_depth_zero_returns_static_evaluation_without_move(self) -> None:
        board = Board.starting_position()

        result = alpha_beta_search(board, depth=0)
        self.assertEqual(result.score, 0)
        self.assertIsNone(result.best_move)
        self.assertEqual(result.nodes_searched, 1)

    def test_quiescence_extends_noisy_depth_zero_positions(self) -> None:
        board = Board.from_fen("4k3/8/8/8/8/8/4q3/3QK3 w - - 0 1")

        result = alpha_beta_search(board, depth=0)
        self.assertGreater(result.score, 0)
        self.assertGreater(result.nodes_searched, 1)

    def test_search_prefers_winning_material(self) -> None:
        board = Board.from_fen("4k3/8/8/8/8/8/4q3/3QK3 w - - 0 1")

        result = alpha_beta_search(board, depth=1)
        self.assertIn(result.best_move, {Move("d1", "e2"), Move("e1", "e2")})
        self.assertGreater(result.score, 0)
        self.assertGreater(result.nodes_searched, 0)

    def test_search_finds_mate_in_one(self) -> None:
        board = Board.from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")

        result = alpha_beta_search(board, depth=1)
        self.assertEqual(result.best_move, Move("g6", "g7"))
        self.assertEqual(choose_move(board, depth=1), Move("g6", "g7"))
        self.assertEqual(result.score, CHECKMATE_SCORE - 1)

    def test_ordered_moves_puts_profitable_capture_before_quiet_moves(self) -> None:
        board = Board.from_fen("4k3/8/8/8/8/8/4q3/3QK3 w - - 0 1")

        moves = ordered_moves(board)
        self.assertEqual(moves[0].to_square, "e2")

    def test_ordered_moves_puts_promotions_first(self) -> None:
        board = Board.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

        moves = ordered_moves(board)
        self.assertEqual(moves[0], Move("a7", "a8", promotion=PieceType.QUEEN))

    def test_ordered_moves_can_force_previous_best_move_first(self) -> None:
        board = Board.starting_position()
        preferred = Move("h2", "h3")

        moves = ordered_moves(board, preferred_move=preferred)
        self.assertEqual(moves[0], preferred)

    def test_iterative_deepening_reuses_previous_best_move(self) -> None:
        board = Board.from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")

        result = iterative_deepening_search(board, max_depth=2)
        self.assertEqual(result.best_move, Move("g6", "g7"))
        self.assertEqual(choose_move(board, depth=2), Move("g6", "g7"))
        self.assertGreater(result.nodes_searched, 1)

    def test_search_populates_transposition_table(self) -> None:
        board = Board.from_fen("4k3/8/8/8/8/8/4q3/3QK3 w - - 0 1")
        table = TranspositionTable()

        result = alpha_beta_search(board, depth=2, transposition_table=table)

        entry = table.lookup(board.zobrist_hash)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.depth, 2)
        self.assertEqual(entry.score, result.score)
        self.assertEqual(entry.bound, BoundType.EXACT)
        self.assertEqual(entry.best_move, result.best_move)

    def test_search_reuses_exact_transposition_entry_on_repeat_search(self) -> None:
        board = Board.from_fen("4k3/8/8/8/8/8/4q3/3QK3 w - - 0 1")
        table = TranspositionTable()

        first_result = alpha_beta_search(board, depth=2, transposition_table=table)
        second_result = alpha_beta_search(board, depth=2, transposition_table=table)

        self.assertEqual(second_result.score, first_result.score)
        self.assertEqual(second_result.best_move, first_result.best_move)
        self.assertEqual(second_result.nodes_searched, 1)


if __name__ == "__main__":
    unittest.main()
