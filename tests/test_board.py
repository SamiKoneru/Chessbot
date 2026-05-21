import unittest

from bot.board import Board
from bot.enums import Color, PieceType
from bot.move import Move
from bot.piece import Piece
from bot.zobrist import compute_zobrist_hash


class BoardTests(unittest.TestCase):
    def test_starting_position_round_trips_to_fen(self) -> None:
        board = Board.starting_position()
        self.assertEqual(
            board.to_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )

    def test_basic_pawn_move_updates_turn_and_en_passant(self) -> None:
        board = Board.starting_position()
        board.apply_move(Move("e2", "e4"))

        self.assertEqual(board.piece_at("e4"), Piece(Color.WHITE, PieceType.PAWN))
        self.assertIsNone(board.piece_at("e2"))
        self.assertIs(board.state.side_to_move, Color.BLACK)
        self.assertEqual(board.state.en_passant_target, "e3")

    def test_castling_repositions_rook(self) -> None:
        board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        board.apply_move(Move("e1", "g1", is_castling=True))

        self.assertEqual(board.piece_at("g1"), Piece(Color.WHITE, PieceType.KING))
        self.assertEqual(board.piece_at("f1"), Piece(Color.WHITE, PieceType.ROOK))
        self.assertIsNone(board.piece_at("h1"))

    def test_knight_attack_detection(self) -> None:
        board = Board.from_fen("4k3/8/8/3n4/8/8/8/4K3 w - - 0 1")

        self.assertTrue(board.is_square_attacked("f4", by_color=Color.BLACK))
        self.assertTrue(board.is_square_attacked("e3", by_color=Color.BLACK))
        self.assertFalse(board.is_square_attacked("d4", by_color=Color.BLACK))

    def test_slider_attacks_stop_after_first_blocker(self) -> None:
        board = Board.from_fen("4k3/8/8/8/3R4/3P4/8/4K3 w - - 0 1")

        rook_attacks = board.attacks_from("d4")
        self.assertIn("d3", rook_attacks)
        self.assertNotIn("d2", rook_attacks)
        self.assertIn("d8", rook_attacks)

    def test_check_detection_finds_rook_check(self) -> None:
        board = Board.from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")

        self.assertTrue(board.is_in_check(Color.WHITE))
        self.assertFalse(board.is_in_check(Color.BLACK))

    def test_default_check_query_uses_side_to_move(self) -> None:
        board = Board.from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        self.assertTrue(board.is_in_check())

    def test_zobrist_hash_matches_recomputed_starting_position(self) -> None:
        board = Board.starting_position()
        self.assertEqual(board.zobrist_hash, compute_zobrist_hash(board))

    def test_zobrist_hash_changes_with_state_bits(self) -> None:
        white_to_move = Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        black_to_move = Board.from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
        with_castling = Board.from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
        without_castling = Board.from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
        with_en_passant = Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - e3 0 1")
        without_en_passant = Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")

        self.assertNotEqual(white_to_move.zobrist_hash, black_to_move.zobrist_hash)
        self.assertNotEqual(with_castling.zobrist_hash, without_castling.zobrist_hash)
        self.assertNotEqual(with_en_passant.zobrist_hash, without_en_passant.zobrist_hash)

    def test_zobrist_hash_stays_in_sync_after_move_updates(self) -> None:
        cases = [
            (
                Board.starting_position(),
                Move("e2", "e4"),
            ),
            (
                Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
                Move("e1", "g1", is_castling=True),
            ),
            (
                Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
                Move("e5", "d6", is_en_passant=True),
            ),
            (
                Board.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
                Move("a7", "a8", promotion=PieceType.QUEEN),
            ),
        ]

        for board, move in cases:
            board.apply_move(move, validate_legality=True)
            self.assertEqual(board.zobrist_hash, compute_zobrist_hash(board))


if __name__ == "__main__":
    unittest.main()
