import unittest

from bot.board import Board
from bot.evaluation import CHECKMATE_SCORE, evaluate, evaluate_for_side_to_move, material_balance
from bot.enums import Color


class EvaluationTests(unittest.TestCase):
    def test_starting_position_is_equal(self) -> None:
        board = Board.starting_position()

        self.assertEqual(material_balance(board), 0)
        self.assertEqual(evaluate(board, Color.WHITE), 0)
        self.assertEqual(evaluate(board, Color.BLACK), 0)

    def test_material_balance_tracks_simple_piece_advantage(self) -> None:
        board = Board.from_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")

        self.assertEqual(material_balance(board), 900)
        self.assertEqual(evaluate(board, Color.WHITE), 900)
        self.assertEqual(evaluate(board, Color.BLACK), -900)

    def test_checkmate_scores_are_extreme(self) -> None:
        board = Board.from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")

        self.assertEqual(evaluate(board, Color.WHITE), CHECKMATE_SCORE)
        self.assertEqual(evaluate(board, Color.BLACK), -CHECKMATE_SCORE)
        self.assertEqual(evaluate_for_side_to_move(board), -CHECKMATE_SCORE)

    def test_stalemate_is_scored_as_zero(self) -> None:
        board = Board.from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")

        self.assertEqual(evaluate(board, Color.WHITE), 0)
        self.assertEqual(evaluate(board, Color.BLACK), 0)
        self.assertEqual(evaluate_for_side_to_move(board), 0)


if __name__ == "__main__":
    unittest.main()
