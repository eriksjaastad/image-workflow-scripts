#!/usr/bin/env python3
"""
Comprehensive tests for prompt extraction functions in character_processor.py
Tests ethnicity, age, body_type, and hair_color extraction with whole-prompt accuracy.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import extraction functions
try:
    # Import from scripts.02_character_processor (with number)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "character_processor",
        Path(__file__).parent.parent / "02_character_processor.py",
    )
    character_processor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(character_processor)

    extract_ethnicity_from_prompt = character_processor.extract_ethnicity_from_prompt
    extract_age_from_prompt = character_processor.extract_age_from_prompt
    extract_body_type_from_prompt = character_processor.extract_body_type_from_prompt
    extract_hair_color_from_prompt = character_processor.extract_hair_color_from_prompt
except Exception as e:
    print(f"[!] Failed to import character processor functions: {e}")
    sys.exit(1)


class TestEthnicityExtraction(unittest.TestCase):
    """Test ethnicity extraction from various prompt formats."""

    def test_single_word_ethnicities(self):
        """Test single-word ethnicity detection with word boundaries."""
        test_cases = [
            ("A beautiful latina woman smiling", "latina"),
            ("Asian woman in traditional dress", "asian"),
            ("Black professional in office setting", "black"),
            ("White woman with blonde hair", "white"),
            ("Indian woman wearing sari", "indian"),
        ]

        for prompt, expected in test_cases:
            with self.subTest(prompt=prompt):
                result = extract_ethnicity_from_prompt(prompt.lower())
                self.assertEqual(
                    result, expected, f"Failed to extract '{expected}' from: {prompt}"
                )

    def test_multi_word_ethnicities(self):
        """Test multi-word ethnicity phrases."""
        test_cases = [
            ("A middle eastern woman in hijab", "middle_eastern"),
            ("Beautiful mixed race model", "mixed_race"),
            ("South asian woman smiling", "south_asian"),
        ]

        for prompt, expected in test_cases:
            with self.subTest(prompt=prompt):
                result = extract_ethnicity_from_prompt(prompt.lower())
                self.assertEqual(
                    result, expected, f"Failed to extract '{expected}' from: {prompt}"
                )

    def test_ethnicity_at_end_of_prompt(self):
        """Test that ethnicity is found even at the end of a long prompt."""
        prompt = "A beautiful woman with long flowing hair, perfect smile, wearing elegant dress, professional photoshoot, studio lighting, latina"
        result = extract_ethnicity_from_prompt(prompt.lower())
        self.assertEqual(result, "latina", "Should find ethnicity at end of prompt")

    def test_ethnicity_in_middle_of_prompt(self):
        """Test that ethnicity is found in the middle of prompt."""
        prompt = "Photorealistic portrait, asian woman with elegant makeup, beautiful smile, professional lighting"
        result = extract_ethnicity_from_prompt(prompt.lower())
        self.assertEqual(result, "asian", "Should find ethnicity in middle of prompt")

    def test_no_ethnicity(self):
        """Test that None is returned when no ethnicity found."""
        prompt = "A beautiful woman with long hair and a smile"
        result = extract_ethnicity_from_prompt(prompt.lower())
        self.assertIsNone(result, "Should return None when no ethnicity found")

    def test_word_boundaries(self):
        """Test that word boundaries prevent false matches."""
        # 'in' should not match 'indian'
        prompt = "A woman standing in the park"
        result = extract_ethnicity_from_prompt(prompt.lower())
        self.assertIsNone(result, "'in' should not match 'indian'")


class TestAgeExtraction(unittest.TestCase):
    """Test age extraction from various prompt formats."""

    def test_years_old_pattern(self):
        """Test 'X years old' pattern."""
        test_cases = [
            ("A 25 years old woman", "25_years_old"),
            ("Beautiful 30 year old latina", "30_years_old"),
            ("42 years old professional", "42_years_old"),
        ]

        for prompt, expected in test_cases:
            with self.subTest(prompt=prompt):
                result = extract_age_from_prompt(prompt.lower())
                self.assertEqual(
                    result, expected, f"Failed to extract '{expected}' from: {prompt}"
                )

    def test_in_her_decades_pattern(self):
        """Test 'in her/his X decades' pattern."""
        test_cases = [
            ("Woman in her mid 20s", "mid_20s"),
            ("Man in his early 30s", "early_30s"),
            ("In her late 40s", "late_40s"),
            ("In his 30s", "30s"),
        ]

        for prompt, expected in test_cases:
            with self.subTest(prompt=prompt):
                result = extract_age_from_prompt(prompt.lower())
                self.assertEqual(
                    result, expected, f"Failed to extract '{expected}' from: {prompt}"
                )

    def test_simple_decades_pattern(self):
        """Test simple 'early/mid/late X0s' pattern."""
        test_cases = [
            ("Beautiful early 20s latina", "early_20s"),
            ("Professional mid 30s woman", "mid_30s"),
            ("Mature late 40s executive", "late_40s"),
        ]

        for prompt, expected in test_cases:
            with self.subTest(prompt=prompt):
                result = extract_age_from_prompt(prompt.lower())
                self.assertEqual(
                    result, expected, f"Failed to extract '{expected}' from: {prompt}"
                )

    def test_age_at_end_of_prompt(self):
        """Test that age is found even at the end of prompt."""
        prompt = "Beautiful woman with long hair, professional lighting, elegant dress, 28 years old"
        result = extract_age_from_prompt(prompt.lower())
        self.assertEqual(result, "28_years_old", "Should find age at end of prompt")

    def test_no_age(self):
        """Test that None is returned when no age found."""
        prompt = "A beautiful woman with long hair"
        result = extract_age_from_prompt(prompt.lower())
        self.assertIsNone(result, "Should return None when no age found")


class TestBodyTypeExtraction(unittest.TestCase):
    """Test body type extraction from entire prompt."""

    def test_single_word_body_types(self):
        """Test single-word body type detection."""
        test_cases = [
            ("A petite latina woman", "petite"),
            ("Curvy woman in red dress", "curvy"),
            ("Athletic build with toned muscles", "athletic"),
            ("Voluptuous figure in elegant gown", "voluptuous"),
            ("Slim woman with long legs", "slim"),
            ("Busty blonde in tight top", "busty"),
            ("Tall woman in heels", "tall"),
        ]

        for prompt, expected in test_cases:
            with self.subTest(prompt=prompt):
                result = extract_body_type_from_prompt(prompt.lower())
                self.assertEqual(
                    result, expected, f"Failed to extract '{expected}' from: {prompt}"
                )

    def test_multi_word_body_types(self):
        """Test multi-word body type phrases."""
        test_cases = [
            ("Woman with big boobs wearing bikini", "big_boobs"),
            ("Beautiful latina with big tits", "big_tits"),
        ]

        for prompt, expected in test_cases:
            with self.subTest(prompt=prompt):
                result = extract_body_type_from_prompt(prompt.lower())
                self.assertEqual(
                    result, expected, f"Failed to extract '{expected}' from: {prompt}"
                )

    def test_body_type_at_end_of_prompt(self):
        """Test that body type is found at end of long prompt."""
        prompt = "Beautiful latina woman, long black hair, wearing red dress, professional photoshoot, studio lighting, elegant makeup, voluptuous"
        result = extract_body_type_from_prompt(prompt.lower())
        self.assertEqual(result, "voluptuous", "Should find body type at end of prompt")

    def test_body_type_in_middle_of_prompt(self):
        """Test that body type is found in middle of prompt."""
        prompt = "Professional photoshoot, curvy latina woman with long hair, elegant dress, perfect lighting"
        result = extract_body_type_from_prompt(prompt.lower())
        self.assertEqual(result, "curvy", "Should find body type in middle of prompt")

    def test_no_body_type(self):
        """Test that None is returned when no body type found."""
        prompt = "A beautiful woman with long hair and a smile"
        result = extract_body_type_from_prompt(prompt.lower())
        self.assertIsNone(result, "Should return None when no body type found")

    def test_priority_multi_word_over_single(self):
        """Test that multi-word phrases are matched before single words."""
        # "big boobs" should be matched as whole phrase, not just "big"
        prompt = "Woman with big boobs"
        result = extract_body_type_from_prompt(prompt.lower())
        self.assertEqual(result, "big_boobs", "Should match 'big boobs' as phrase")


class TestHairColorExtraction(unittest.TestCase):
    """Test hair color extraction from entire prompt."""

    def test_single_word_hair_colors(self):
        """Test single-word hair color detection."""
        test_cases = [
            ("Beautiful blonde woman smiling", "blonde"),
            ("Elegant brunette in evening gown", "brunette"),
            ("Fiery redhead with curls", "redhead"),
            ("Platinum hair woman", "platinum"),
        ]

        for prompt, expected in test_cases:
            with self.subTest(prompt=prompt):
                result = extract_hair_color_from_prompt(prompt.lower())
                self.assertEqual(
                    result, expected, f"Failed to extract '{expected}' from: {prompt}"
                )

    def test_multi_word_hair_colors(self):
        """Test multi-word hair color phrases."""
        test_cases = [
            ("Woman with black hair", "black_hair"),
            ("Beautiful latina with brown hair", "brown_hair"),
            ("Elegant woman with red hair flowing", "red_hair"),
            ("Silver hair woman in elegant dress", "silver_hair"),
            ("Woman with gray hair smiling", "gray_hair"),
        ]

        for prompt, expected in test_cases:
            with self.subTest(prompt=prompt):
                result = extract_hair_color_from_prompt(prompt.lower())
                self.assertEqual(
                    result, expected, f"Failed to extract '{expected}' from: {prompt}"
                )

    def test_hair_color_at_end_of_prompt(self):
        """Test that hair color is found at end of prompt."""
        prompt = "Beautiful woman in elegant dress, professional lighting, perfect makeup, long flowing blonde"
        result = extract_hair_color_from_prompt(prompt.lower())
        self.assertEqual(result, "blonde", "Should find hair color at end of prompt")

    def test_no_hair_color(self):
        """Test that None is returned when no hair color found."""
        prompt = "A beautiful woman with a smile"
        result = extract_hair_color_from_prompt(prompt.lower())
        self.assertIsNone(result, "Should return None when no hair color found")


class TestRealWorldPrompts(unittest.TestCase):
    """Test with real-world complex prompts."""

    def test_complex_prompt_all_descriptors(self):
        """Test extraction from a complex prompt with multiple descriptors."""
        prompt = "Professional photoshoot of a voluptuous latina woman in her mid 20s, long black hair, wearing elegant red dress, studio lighting, perfect makeup, confident pose"

        ethnicity = extract_ethnicity_from_prompt(prompt.lower())
        age = extract_age_from_prompt(prompt.lower())
        body_type = extract_body_type_from_prompt(prompt.lower())
        hair_color = extract_hair_color_from_prompt(prompt.lower())

        self.assertEqual(ethnicity, "latina")
        self.assertEqual(age, "mid_20s")
        self.assertEqual(body_type, "voluptuous")
        self.assertEqual(hair_color, "black_hair")

    def test_descriptors_scattered_throughout(self):
        """Test that descriptors are found regardless of position."""
        prompt = "In her early 30s, an asian woman with long brown hair walks confidently, her athletic build evident in her movement, professional photography"

        ethnicity = extract_ethnicity_from_prompt(prompt.lower())
        age = extract_age_from_prompt(prompt.lower())
        body_type = extract_body_type_from_prompt(prompt.lower())
        hair_color = extract_hair_color_from_prompt(prompt.lower())

        self.assertEqual(ethnicity, "asian")
        self.assertEqual(age, "early_30s")
        self.assertEqual(body_type, "athletic")
        self.assertEqual(hair_color, "brown_hair")

    def test_very_long_prompt(self):
        """Test extraction from very long prompt (descriptors at various positions)."""
        prompt = (
            "A breathtaking professional photograph captures a stunning petite latina woman, "
            "her radiant smile illuminating the frame as soft natural light filters through sheer curtains, "
            "creating a warm, inviting atmosphere. She's dressed in an elegant flowing white dress, "
            "the fabric cascading gracefully around her silhouette. Her posture exudes confidence and poise, "
            "while her expressive eyes convey warmth and authenticity. The composition employs a shallow depth of field, "
            "beautifully blurring the background to emphasize her captivating presence. Shot in early 20s aesthetic style."
        )

        ethnicity = extract_ethnicity_from_prompt(prompt.lower())
        age = extract_age_from_prompt(prompt.lower())
        body_type = extract_body_type_from_prompt(prompt.lower())

        self.assertEqual(ethnicity, "latina", "Should find ethnicity in long prompt")
        self.assertEqual(age, "early_20s", "Should find age at end of long prompt")
        self.assertEqual(body_type, "petite", "Should find body type in long prompt")


def run_tests():
    """Run all tests and display results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEthnicityExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestAgeExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestBodyTypeExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestHairColorExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestRealWorldPrompts))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
