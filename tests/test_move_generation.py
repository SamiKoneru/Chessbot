import unittest

from bot.board import Board
from bot.move import Move


class MoveGenerationTests(unittest.TestCase):
    def test_starting_position_has_twenty_legal_moves(self) -> None:
        board = Board.starting_position()

        legal_moves = board.legal_moves()
        self.assertEqual(len(legal_moves), 20)
        self.assertIn(Move("e2", "e4"), legal_moves)
        self.assertIn(Move("g1", "f3"), legal_moves)

    def test_pinned_piece_move_is_filtered_out_of_legal_moves(self) -> None:
        board = Board.from_fen("4r3/8/8/8/8/8/4R3/4K3 w - - 0 1")

        legal_moves = board.legal_moves()
        self.assertNotIn(Move("e2", "a2"), legal_moves)
        self.assertIn(Move("e2", "e8"), legal_moves)

    def test_en_passant_is_generated_when_available(self) -> None:
        board = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")

        self.assertIn(Move("e5", "d6", is_en_passant=True), board.legal_moves())
        self.assertTrue(board.is_legal_move(Move("e5", "d6")))

    def test_castling_is_generated_when_path_is_clear_and_safe(self) -> None:
        board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

        legal_moves = board.legal_moves()
        self.assertIn(Move("e1", "g1", is_castling=True), legal_moves)
        self.assertIn(Move("e1", "c1", is_castling=True), legal_moves)
        self.assertTrue(board.is_legal_move(Move("e1", "g1")))

    def test_castling_through_attack_is_not_legal(self) -> None:
        board = Board.from_fen("r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1")

        legal_moves = board.legal_moves()
        self.assertNotIn(Move("e1", "g1", is_castling=True), legal_moves)
        self.assertIn(Move("e1", "c1", is_castling=True), legal_moves)

    def test_apply_move_can_enforce_legality(self) -> None:
        board = Board.from_fen("4r3/8/8/8/8/8/4R3/4K3 w - - 0 1")

        with self.assertRaises(ValueError):
            board.apply_move(Move("e2", "a2"), validate_legality=True)

        board.apply_move(Move("e2", "e8"), validate_legality=True)
        self.assertEqual(board.to_fen(), "4R3/8/8/8/8/8/8/4K3 b - - 0 1")

    def test_king_capture_is_not_generated_or_applied(self) -> None:
        board = Board.from_fen("k7/1Q6/2K5/8/8/8/8/8 w - - 0 1")

        illegal_capture = Move("b7", "a8")
        self.assertNotIn(illegal_capture, board.pseudo_legal_moves())
        self.assertNotIn(illegal_capture, board.legal_moves())

        with self.assertRaises(ValueError):
            board.apply_move(illegal_capture)

    def test_checkmate_detection_uses_check_plus_no_legal_moves(self) -> None:
        board = Board.from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")

        self.assertTrue(board.is_in_check())
        self.assertFalse(board.has_legal_moves())
        self.assertTrue(board.is_checkmate())
        self.assertFalse(board.is_stalemate())
        self.assertTrue(board.is_game_over())

    def test_stalemate_detection_uses_no_legal_moves_without_check(self) -> None:
        board = Board.from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")

        self.assertFalse(board.is_in_check())
        self.assertFalse(board.has_legal_moves())
        self.assertFalse(board.is_checkmate())
        self.assertTrue(board.is_stalemate())
        self.assertTrue(board.is_game_over())


if __name__ == "__main__":
    unittest.main()
