import unittest
from src.zeroband.sanskrit_poetry_env import SanskritPoetryEnv

class TestSanskritPoetryEnv(unittest.TestCase):
    def setUp(self):
        self.env = SanskritPoetryEnv(
            dataset_name="nippun-live/sanskrit-poetry-requests",
            max_steps=2
        )

    def test_reset(self):
        req, info = self.env.reset()
        self.assertIsInstance(req, str)
        self.assertIn("Write a poem about", req)
        self.assertEqual(info["expected_meter"], "Anushtup")

    def test_step_incorrect_poem(self):
        self.env.reset()
        r, done, info = self.env.step("foo bar baz")
        self.assertEqual(r, -1.0)
        self.assertFalse(done)
        self.assertIn("identified_meter", info)

    def test_step_correct_meter(self):
        # Skip the first (Anushtup) request
        self.env.reset()
        self.env.step("dummy text")

        # Now test the known Praharṣiṇī couplet
        stanza = (
            "निर्दिष्टाङ् कुलपतिना स पर्णशालाम् अध्यास्य प्रयतपरिग्रहद्वितीयः ।\n"
            "तच्छिष्याध्ययननिवेदितावसानां सव्ँविष्टः कुशशयने निशान् निनाय ॥"
        )
        reward, done, info = self.env.step(stanza)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["identified_meter"], "Praharṣiṇī")

if __name__ == "__main__":
    unittest.main()
