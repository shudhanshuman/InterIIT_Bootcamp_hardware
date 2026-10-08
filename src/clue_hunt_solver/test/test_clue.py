"""Unit tests for clue.py - chain rules from final_plan.md §2.1 / §4.3.

Fixtures are the decoded practice-world textures (offline ground truth from
§2.1); they double as regression vectors for the token chain.
"""
import unittest

from clue_hunt_solver.clue import (ChainValidator, START_TOKEN, Verdict,
                                   parse_clue, token_for)

B1 = 'HUNT:1:7196:GOTO 1.5 -3.9'
B2 = 'HUNT:2:7F49:PILLAR RED'
B3 = 'HUNT:3:BCB8:BETWEEN BLUE GREEN 0.59'
B4 = 'HUNT:4:EC9E:REL 5.08 -0.65'
B5 = 'HUNT:5:1756:TREASURE REL 0.98 3.15'
D7 = 'HUNT:7:70DD:GOTO 0.0 0.0'
X4 = 'HUNT:4:07A8:REL 2.00 0.00'


class TestToken(unittest.TestCase):
    def test_start_token(self):
        self.assertEqual(token_for('START'), '7196')
        self.assertEqual(START_TOKEN, '7196')

    def test_fixture_chain_is_self_consistent(self):
        texts = [B1, B2, B3, B4, B5]
        for prev, cur in zip(['START'] + texts[:-1], texts):
            with self.subTest(cur=cur):
                self.assertEqual(token_for(prev), cur.split(':')[2])


class TestParse(unittest.TestCase):
    def test_goto(self):
        c = parse_clue(B1)
        self.assertEqual((c.id, c.token, c.verb, c.args, c.treasure),
                         (1, '7196', 'GOTO', (1.5, -3.9), False))

    def test_pillar(self):
        c = parse_clue(B2)
        self.assertEqual((c.verb, c.args), ('PILLAR', ('RED',)))

    def test_between(self):
        c = parse_clue(B3)
        self.assertEqual((c.verb, c.args), ('BETWEEN', ('BLUE', 'GREEN', 0.59)))

    def test_rel(self):
        c = parse_clue(B4)
        self.assertEqual((c.verb, c.args), ('REL', (5.08, -0.65)))

    def test_treasure_rel(self):
        c = parse_clue(B5)
        self.assertTrue(c.treasure)
        self.assertEqual((c.verb, c.args), ('REL', (0.98, 3.15)))

    def test_malformed(self):
        for bad in ('', 'HUNT:1:7196', 'hunt:1:7196:GOTO 1 2',
                    'HUNT:x:7196:GOTO 1 2', 'HUNT:1:zzzz:GOTO 1 2',
                    'HUNT:1:7196:GOTO 1', 'HUNT:1:7196:FLY 1 2',
                    'HUNT:5:1756:TREASURE GOTO 1 2',
                    'GOTO 1 2'):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_clue(bad))


class TestChain(unittest.TestCase):
    def test_full_practice_chain(self):
        v = ChainValidator()
        for text in (B1, B2, B3, B4, B5):
            verdict, clue = v.validate(text)
            with self.subTest(text=text):
                self.assertEqual(verdict, Verdict.VALID)
                self.assertEqual(clue.raw, text)
        self.assertEqual(v.accepted, 5)
        self.assertEqual(v.anchor_id, 1)

    def test_root_adopts_anchor_id(self):
        v = ChainValidator()
        verdict, clue = v.validate('HUNT:9:7196:GOTO 1 2')
        self.assertEqual(verdict, Verdict.VALID)
        self.assertEqual(v.anchor_id, 9)
        # next expected id is now 10; id 11 -> decoy, id 10 + bad token -> look-alike
        self.assertEqual(v.validate('HUNT:11:0000:GOTO 1 2')[0], Verdict.DECOY)
        self.assertEqual(v.validate('HUNT:10:0000:GOTO 1 2')[0], Verdict.LOOKALIKE)
        self.assertEqual(v.accepted, 1)

    def test_decoy_d7_rejected_state_unchanged(self):
        v = ChainValidator()
        v.validate(B1)
        verdict, clue = v.validate(D7)
        self.assertEqual(verdict, Verdict.DECOY)
        self.assertEqual(clue.id, 7)
        self.assertEqual(v.accepted, 1)
        self.assertEqual(v.validate(B2)[0], Verdict.VALID)

    def test_lookalike_x4_rejected_then_real_b4_ok(self):
        v = ChainValidator()
        for text in (B1, B2, B3):
            self.assertEqual(v.validate(text)[0], Verdict.VALID)
        verdict, clue = v.validate(X4)
        self.assertEqual(verdict, Verdict.LOOKALIKE)
        self.assertEqual(clue.id, 4)
        self.assertEqual(v.accepted, 3)
        self.assertEqual(v.validate(B4)[0], Verdict.VALID)
        self.assertEqual(v.validate(B5)[0], Verdict.VALID)

    def test_out_of_order_is_decoy(self):
        v = ChainValidator()
        v.validate(B1)
        self.assertEqual(v.validate(B3)[0], Verdict.DECOY)
        self.assertEqual(v.validate(B2)[0], Verdict.VALID)

    def test_pre_root_non_root_is_decoy(self):
        v = ChainValidator()
        self.assertEqual(v.validate(B2)[0], Verdict.DECOY)
        self.assertEqual(v.accepted, 0)
        self.assertEqual(v.validate(B1)[0], Verdict.VALID)

    def test_malformed_is_invalid_and_state_unchanged(self):
        v = ChainValidator()
        v.validate(B1)
        self.assertEqual(v.validate('garbage')[0], Verdict.INVALID)
        self.assertEqual(v.accepted, 1)
        self.assertEqual(v.validate(B2)[0], Verdict.VALID)

    def test_no_recasing(self):
        v = ChainValidator()
        self.assertEqual(v.validate(B1.lower())[0], Verdict.INVALID)


if __name__ == '__main__':
    unittest.main()