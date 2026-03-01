"""Unit tests for TUI format-based evaluators.

Covers:
- harmony_format_score: Harmony token detection
- tool_call_score: Valid/invalid/absent tool calls
- completeness_score: Truncation detection
- _detect_repetition: Repetition loop detection
- compute_execution_reward: Composite reward + penalties
- rank_solutions_by_execution: Ranking order and details dict
- TUIEvaluator: Protocol compliance and registry registration
- generate_tui_grpo_tasks.py: extract_first_user_message and generate_tasks
"""
from __future__ import annotations

import sys
import os
import pytest

# ---------------------------------------------------------------------------
# Path setup (mirrors conftest.py behavior for scripts/ modules)
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline_lib.tui_evaluators import (
    TUIEvaluator,
    _detect_repetition,
    compute_execution_reward,
    completeness_score,
    harmony_format_score,
    rank_solutions_by_execution,
    tool_call_score,
)


# ---------------------------------------------------------------------------
# Fixtures: sample Harmony-formatted completions
# ---------------------------------------------------------------------------

FULL_HARMONY = (
    "<|developer|>\nYou are an agent.\n"
    "<|user|>\nList the files in src/.\n"
    "<|assistant|>\nSure, let me check.\n"
    "<|endoftext|>"
)

FULL_HARMONY_WITH_TOOL = (
    "<|developer|>\nYou are an agent.\n"
    "<|user|>\nRun the tests.\n"
    "<|tool_call|>\n"
    '{"id": "1", "name": "run_tests", "arguments": {"cmd": "cargo test"}}\n'
    "<|tool_result|>\nAll tests passed.\n"
    "<|assistant|>\nAll tests passed.\n"
    "<|endoftext|>"
)

TRUNCATED_CODE_FENCE = (
    "<|assistant|>\nHere is the solution:\n```rust\nfn main() {"
)

TRUNCATED_DANGLING = (
    "<|assistant|>\nI will call the function with arguments: {"
)

EMPTY_RESPONSE = ""

REPETITION_RESPONSE = (
    "This is a repeated phrase. " * 10  # Same phrase repeated many times
)


# ===========================================================================
# TestHarmonyFormatScore
# ===========================================================================

class TestHarmonyFormatScore:
    """Tests for harmony_format_score()."""

    def test_full_score_with_assistant_and_endoftext(self):
        """Should score 0.5 when both <|assistant|> and <|endoftext|> are present."""
        score = harmony_format_score(FULL_HARMONY)
        assert score == pytest.approx(0.5), (
            f"Expected 0.5 for full Harmony response, got {score}"
        )

    def test_partial_score_assistant_only(self):
        """Should score 0.3 when only <|assistant|> is present."""
        text = "<|assistant|>\nHere is my response."
        score = harmony_format_score(text)
        assert score == pytest.approx(0.3), (
            f"Expected 0.3 for assistant-only, got {score}"
        )

    def test_partial_score_endoftext_only(self):
        """Should score 0.2 when only <|endoftext|> is present (no assistant marker)."""
        text = "Some plain text.<|endoftext|>"
        score = harmony_format_score(text)
        assert score == pytest.approx(0.2), (
            f"Expected 0.2 for endoftext-only, got {score}"
        )

    def test_zero_score_for_plain_text(self):
        """Should score 0.0 when no Harmony tokens are present."""
        text = "Hello, world. This is a plain text response."
        score = harmony_format_score(text)
        assert score == pytest.approx(0.0), (
            f"Expected 0.0 for plain text, got {score}"
        )

    def test_alternate_ending_token(self):
        """<|end|> should count as an ending token."""
        text = "<|assistant|>\nDone.<|end|>"
        score = harmony_format_score(text)
        assert score == pytest.approx(0.5), (
            f"Expected 0.5 with <|end|> token, got {score}"
        )


# ===========================================================================
# TestToolCallScore
# ===========================================================================

class TestToolCallScore:
    """Tests for tool_call_score()."""

    def test_no_tool_call_marker_returns_default(self):
        """Pure-text response should get 0.2 (correct not to call a tool)."""
        score = tool_call_score("I will look at this carefully and respond.")
        assert 0.0 < score <= 0.3, (
            f"Expected partial credit for no-tool-call response, got {score}"
        )

    def test_valid_tool_call_with_name_and_json_args_scores_full(self):
        """Valid JSON tool call with name and args should score 0.3."""
        completion = (
            '<|tool_call|>\n'
            '{"id": "1", "name": "read_file", "arguments": {"path": "src/main.rs"}}\n'
        )
        score = tool_call_score(completion)
        assert score == pytest.approx(0.3), (
            f"Expected 0.3 for valid tool call, got {score}"
        )

    def test_tool_call_with_string_args_that_parse_as_json(self):
        """Arguments as a JSON string (not dict) should also validate correctly."""
        completion = (
            '<|tool_call|>\n'
            '{"id": "1", "name": "run_tests", "arguments": "{\\"cmd\\": \\"cargo test\\"}"}\n'
        )
        score = tool_call_score(completion)
        # Should have name (0.1) and valid args (0.2)
        assert score == pytest.approx(0.3), (
            f"Expected 0.3 for tool call with JSON string args, got {score}"
        )

    def test_invalid_json_tool_call_loses_args_credit(self):
        """Malformed JSON in tool call block should lose the 0.2 args credit."""
        completion = (
            '<|tool_call|>\n'
            'NOT VALID JSON\n'
        )
        score = tool_call_score(completion)
        assert score < 0.2, (
            f"Expected < 0.2 for invalid JSON tool call, got {score}"
        )

    def test_tool_call_without_name_field(self):
        """Tool call JSON without a 'name' field should not earn the 0.1 name credit."""
        completion = (
            '<|tool_call|>\n'
            '{"id": "1", "arguments": {"path": "main.rs"}}\n'
        )
        score = tool_call_score(completion)
        # May still earn 0.2 for valid JSON but not 0.1 for name
        assert score <= 0.2, (
            f"Expected <= 0.2 for tool call without name, got {score}"
        )

    def test_custom_reward_config_weight_caps_score(self):
        """tool_call_weight in reward_config should cap the returned score."""
        completion = (
            '<|tool_call|>\n'
            '{"id": "1", "name": "run_tests", "arguments": {}}\n'
        )
        score = tool_call_score(completion, reward_config={"tool_call_weight": 0.1})
        assert score <= 0.1, (
            f"Expected score <= 0.1 with custom cap, got {score}"
        )


# ===========================================================================
# TestCompletenessScore
# ===========================================================================

class TestCompletenessScore:
    """Tests for completeness_score()."""

    def test_clean_response_scores_full(self):
        """A well-formed response without truncation signs should score 0.3."""
        text = "<|assistant|>\nThe answer is 42.<|endoftext|>"
        score = completeness_score(text)
        assert score == pytest.approx(0.3), (
            f"Expected 0.3 for clean response, got {score}"
        )

    def test_unclosed_code_fence_scores_zero(self):
        """Response ending with an open code fence should score 0.0."""
        score = completeness_score(TRUNCATED_CODE_FENCE)
        assert score == pytest.approx(0.0), (
            f"Expected 0.0 for unclosed code fence, got {score}"
        )

    def test_dangling_brace_scores_zero(self):
        """Response ending with { (open brace) should score 0.0."""
        score = completeness_score(TRUNCATED_DANGLING)
        assert score == pytest.approx(0.0), (
            f"Expected 0.0 for dangling brace, got {score}"
        )

    def test_empty_string_scores_zero(self):
        """Empty string should score 0.0."""
        score = completeness_score("")
        assert score == pytest.approx(0.0), (
            f"Expected 0.0 for empty string, got {score}"
        )

    def test_closed_code_fence_does_not_truncate(self):
        """Balanced code fences should not trigger truncation detection."""
        text = (
            "<|assistant|>\nHere is the code:\n"
            "```rust\nfn main() {}\n```\n"
            "That should work.<|endoftext|>"
        )
        score = completeness_score(text)
        assert score == pytest.approx(0.3), (
            f"Expected 0.3 for balanced code fences, got {score}"
        )

    def test_custom_completeness_weight(self):
        """completeness_weight in reward_config should change the full score."""
        text = "<|assistant|>\nDone.<|endoftext|>"
        score = completeness_score(text, reward_config={"completeness_weight": 0.5})
        assert score == pytest.approx(0.5), (
            f"Expected 0.5 with custom weight, got {score}"
        )


# ===========================================================================
# TestDetectRepetition
# ===========================================================================

class TestDetectRepetition:
    """Tests for _detect_repetition()."""

    def test_repeated_chunk_detected(self):
        """A chunk repeated 3+ times should be detected."""
        chunk = "This is a repetitive sentence that fills space. "
        text = chunk * 5
        assert _detect_repetition(text) is True, (
            "Expected repetition to be detected for 5x repeated chunk"
        )

    def test_unique_text_not_flagged(self):
        """Normal, non-repetitive text should not be flagged."""
        text = (
            "The model analyzed the problem carefully. "
            "It considered multiple approaches before deciding. "
            "The final answer incorporates all constraints."
        )
        assert _detect_repetition(text) is False, (
            "Expected no repetition detection for unique prose"
        )

    def test_short_text_not_flagged(self):
        """Text shorter than min_chunk * min_repeats should never be flagged."""
        text = "Short."
        assert _detect_repetition(text) is False, (
            "Short text should not trigger repetition detection"
        )

    def test_two_repeats_below_threshold_not_flagged(self):
        """A chunk repeated only twice should not be flagged (threshold is 3)."""
        chunk = "This chunk appears twice. " * 1
        text = chunk + chunk  # Only 2 repetitions
        # With a 50-char min_chunk, 2 repeats of a ~26-char string is below threshold
        assert _detect_repetition(text, min_chunk=20, min_repeats=3) is False, (
            "Two repetitions should not trigger detection at min_repeats=3"
        )


# ===========================================================================
# TestComputeExecutionReward
# ===========================================================================

class TestComputeExecutionReward:
    """Tests for the composite compute_execution_reward() function."""

    def test_empty_string_returns_empty_penalty(self):
        """Empty input must return the empty penalty (-0.5 default)."""
        reward = compute_execution_reward("")
        assert reward == pytest.approx(-0.5), (
            f"Expected -0.5 for empty input, got {reward}"
        )

    def test_whitespace_only_returns_empty_penalty(self):
        """Whitespace-only input should also return empty penalty."""
        reward = compute_execution_reward("   \n\t  ")
        assert reward == pytest.approx(-0.5), (
            f"Expected -0.5 for whitespace-only input, got {reward}"
        )

    def test_full_harmony_response_scores_high(self):
        """A complete, well-formed Harmony response should score near 1.0."""
        reward = compute_execution_reward(FULL_HARMONY)
        assert reward >= 0.9, (
            f"Expected reward >= 0.9 for full Harmony response, got {reward}"
        )

    def test_full_harmony_with_tool_call_scores_high(self):
        """A complete Harmony response with valid tool call should score near 1.0."""
        reward = compute_execution_reward(FULL_HARMONY_WITH_TOOL)
        assert reward >= 0.8, (
            f"Expected reward >= 0.8 for Harmony with tool call, got {reward}"
        )

    def test_truncated_response_scores_lower(self):
        """A truncated response should score lower than a complete one."""
        full_reward = compute_execution_reward(FULL_HARMONY)
        truncated_reward = compute_execution_reward(TRUNCATED_CODE_FENCE)
        assert truncated_reward < full_reward, (
            f"Truncated ({truncated_reward}) should score below full ({full_reward})"
        )

    def test_reward_is_clamped_to_one(self):
        """Reward must never exceed 1.0."""
        # Construct an artificially perfect completion
        perfect = (
            "<|assistant|>\nPerfect response.\n"
            "<|tool_call|>\n"
            '{"id": "1", "name": "run", "arguments": {}}\n'
            "<|endoftext|>"
        )
        reward = compute_execution_reward(perfect)
        assert reward <= 1.0, f"Reward must be <= 1.0, got {reward}"

    def test_reward_clamped_at_empty_penalty_floor(self):
        """Reward must never go below the empty_penalty value."""
        reward = compute_execution_reward("")
        assert reward >= -0.5, f"Reward must be >= -0.5, got {reward}"

    def test_repetition_reduces_reward(self):
        """A repetitive completion should score lower than a unique one."""
        unique = "<|assistant|>\nHere is the answer.<|endoftext|>"
        repetitive = (
            "<|assistant|>\n"
            + "The answer is definitely correct and I am sure about it. " * 10
            + "<|endoftext|>"
        )
        unique_reward = compute_execution_reward(unique)
        repetitive_reward = compute_execution_reward(repetitive)
        assert repetitive_reward <= unique_reward, (
            f"Repetitive ({repetitive_reward}) should not score above unique ({unique_reward})"
        )

    def test_custom_empty_penalty(self):
        """Custom empty_penalty in reward_config should override default -0.5."""
        reward = compute_execution_reward("", reward_config={"empty_penalty": -1.0})
        assert reward == pytest.approx(-1.0), (
            f"Expected custom empty_penalty -1.0, got {reward}"
        )

    def test_tool_calls_argument_is_ignored(self):
        """tool_calls kwarg is accepted but does not change scoring for TUI."""
        reward_without = compute_execution_reward(FULL_HARMONY, tool_calls=None)
        reward_with = compute_execution_reward(
            FULL_HARMONY,
            tool_calls=[{"name": "ignored", "arguments": "{}"}],
        )
        assert reward_without == pytest.approx(reward_with), (
            "tool_calls kwarg should have no effect on TUI scoring"
        )


# ===========================================================================
# TestRankSolutionsByExecution
# ===========================================================================

class TestRankSolutionsByExecution:
    """Tests for rank_solutions_by_execution()."""

    def test_best_solution_ranked_first(self):
        """The solution with highest score must appear first."""
        solutions = [
            TRUNCATED_CODE_FENCE,  # Low score (truncated)
            FULL_HARMONY,           # High score (complete Harmony)
            "",                     # Lowest score (empty)
        ]
        ranked = rank_solutions_by_execution(solutions)
        assert ranked[0][0] == FULL_HARMONY, (
            "Full Harmony response should be ranked first"
        )

    def test_empty_solution_ranked_last(self):
        """Empty completion should be ranked last."""
        solutions = [FULL_HARMONY, TRUNCATED_CODE_FENCE, ""]
        ranked = rank_solutions_by_execution(solutions)
        assert ranked[-1][0] == "", (
            "Empty completion should be ranked last"
        )

    def test_returns_all_solutions(self):
        """All input solutions should appear in the ranked output."""
        solutions = [FULL_HARMONY, TRUNCATED_CODE_FENCE, ""]
        ranked = rank_solutions_by_execution(solutions)
        assert len(ranked) == 3, (
            f"Expected 3 ranked solutions, got {len(ranked)}"
        )

    def test_details_dict_has_required_keys(self):
        """Details dict in ranked output must contain all required boolean keys."""
        ranked = rank_solutions_by_execution([FULL_HARMONY])
        _, score, details = ranked[0]

        required_keys = {
            "has_assistant_marker",
            "has_ending_token",
            "has_tool_calls",
            "tool_calls_valid",
            "response_complete",
            "has_repetition",
        }
        assert required_keys.issubset(set(details.keys())), (
            f"Missing details keys: {required_keys - set(details.keys())}"
        )

    def test_full_harmony_details_correct(self):
        """Full Harmony response details should reflect positive indicators."""
        ranked = rank_solutions_by_execution([FULL_HARMONY])
        _, _, details = ranked[0]
        assert details["has_assistant_marker"] is True
        assert details["has_ending_token"] is True
        assert details["response_complete"] is True
        assert details["has_repetition"] is False

    def test_scores_are_floats(self):
        """All scores in ranked output must be floats."""
        ranked = rank_solutions_by_execution([FULL_HARMONY, ""])
        for _, score, _ in ranked:
            assert isinstance(score, float), (
                f"Score {score!r} is not a float"
            )

    def test_tests_code_kwarg_accepted_and_ignored(self):
        """tests_code kwarg must be accepted for protocol compatibility."""
        ranked = rank_solutions_by_execution(
            [FULL_HARMONY],
            tests_code="# irrelevant for TUI",
        )
        assert len(ranked) == 1, "Should still return 1 result with tests_code kwarg"


# ===========================================================================
# TestTUIEvaluatorClass
# ===========================================================================

class TestTUIEvaluatorClass:
    """Tests for the TUIEvaluator class (registry integration + protocol)."""

    def test_tui_registered_in_evaluators(self):
        """'tui' must be registered in the EVALUATORS dict after import."""
        from pipeline_lib.evaluator_dispatch import EVALUATORS
        assert "tui" in EVALUATORS, (
            "TUIEvaluator was not registered. "
            "Check that tui_evaluators.py uses @register_evaluator('tui')."
        )

    def test_get_evaluator_tui_returns_instance(self):
        """get_evaluator('tui') must return a TUIEvaluator instance."""
        from pipeline_lib.evaluator_dispatch import get_evaluator
        evaluator = get_evaluator("tui")
        assert isinstance(evaluator, TUIEvaluator), (
            f"Expected TUIEvaluator instance, got {type(evaluator)}"
        )

    def test_tui_evaluator_implements_protocol_methods(self):
        """TUIEvaluator must have compute_execution_reward and rank_solutions_by_execution."""
        evaluator = TUIEvaluator()
        assert hasattr(evaluator, "compute_execution_reward")
        assert hasattr(evaluator, "rank_solutions_by_execution")
        assert callable(evaluator.compute_execution_reward)
        assert callable(evaluator.rank_solutions_by_execution)

    def test_tui_evaluator_compute_reward_returns_float(self):
        """compute_execution_reward must return a float."""
        evaluator = TUIEvaluator()
        result = evaluator.compute_execution_reward(FULL_HARMONY)
        assert isinstance(result, float), (
            f"compute_execution_reward returned {type(result)}, expected float"
        )

    def test_tui_evaluator_rank_returns_list_of_tuples(self):
        """rank_solutions_by_execution must return a list of 3-tuples."""
        evaluator = TUIEvaluator()
        ranked = evaluator.rank_solutions_by_execution([FULL_HARMONY, ""])
        assert isinstance(ranked, list)
        assert len(ranked) == 2
        for item in ranked:
            assert len(item) == 3, f"Each item must be a 3-tuple, got {item!r}"

    def test_tui_evaluator_each_call_returns_new_instance(self):
        """get_evaluator should return a fresh instance on each call."""
        from pipeline_lib.evaluator_dispatch import get_evaluator
        a = get_evaluator("tui")
        b = get_evaluator("tui")
        assert a is not b, "get_evaluator should instantiate a new object each call"

    def test_dispatch_compute_reward_routes_to_tui(self):
        """compute_execution_reward(language='tui') must route to TUIEvaluator."""
        from pipeline_lib.evaluator_dispatch import compute_execution_reward as dispatch_cer
        from unittest.mock import MagicMock, patch

        mock_evaluator = MagicMock(spec=TUIEvaluator)
        mock_evaluator.compute_execution_reward.return_value = 0.75

        with patch(
            "pipeline_lib.evaluator_dispatch.get_evaluator",
            return_value=mock_evaluator,
        ) as mock_get:
            result = dispatch_cer(FULL_HARMONY, language="tui")

        mock_get.assert_called_once_with("tui")
        assert result == pytest.approx(0.75)


# ===========================================================================
# TestExtractFirstUserMessage (generate_tui_grpo_tasks.py)
# ===========================================================================

class TestExtractFirstUserMessage:
    """Tests for the Harmony parser in generate_tui_grpo_tasks.py."""

    @pytest.fixture(autouse=True)
    def import_extractor(self):
        """Import extract_first_user_message from the script."""
        # The script lives in scripts/ — import directly via path manipulation
        import importlib.util
        script_path = os.path.join(SCRIPTS_DIR, "generate_tui_grpo_tasks.py")
        spec = importlib.util.spec_from_file_location(
            "generate_tui_grpo_tasks", script_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.extract = mod.extract_first_user_message
        self.generate_tasks = mod.generate_tasks

    def test_extracts_user_message_from_harmony(self):
        """Should extract the user turn from a well-formed Harmony string."""
        text = (
            "<|developer|>\nYou are an agent.\n"
            "<|user|>\nFix the failing test.\n"
            "<|assistant|>\nLet me look at it.\n"
            "<|endoftext|>"
        )
        result = self.extract(text)
        assert result == "Fix the failing test.", (
            f"Expected 'Fix the failing test.', got {result!r}"
        )

    def test_returns_none_for_text_without_user_token(self):
        """Should return None when no <|user|> token is present."""
        text = "<|assistant|>\nSure, here it is.<|endoftext|>"
        result = self.extract(text)
        assert result is None, (
            f"Expected None for text without user token, got {result!r}"
        )

    def test_returns_none_for_empty_string(self):
        """Should return None for empty input."""
        result = self.extract("")
        assert result is None, "Expected None for empty input"

    def test_returns_none_for_empty_user_turn(self):
        """Should return None when user turn has no content."""
        text = "<|user|><|assistant|>\nResponse.<|endoftext|>"
        result = self.extract(text)
        assert result is None, (
            f"Expected None for empty user turn, got {result!r}"
        )

    def test_extracts_first_user_message_when_multiple_turns(self):
        """Should return only the first user message in a multi-turn conversation."""
        text = (
            "<|user|>\nFirst question.\n"
            "<|assistant|>\nFirst answer.\n"
            "<|user|>\nSecond question.\n"
            "<|assistant|>\nSecond answer.\n"
            "<|endoftext|>"
        )
        result = self.extract(text)
        assert result == "First question.", (
            f"Expected 'First question.', got {result!r}"
        )

    def test_strips_whitespace_from_extracted_message(self):
        """Should strip leading/trailing whitespace from the extracted message."""
        text = "<|user|>\n  Implement a sort function.  \n<|assistant|>\nOK.\n"
        result = self.extract(text)
        assert result == "Implement a sort function.", (
            f"Expected trimmed message, got {result!r}"
        )


# ===========================================================================
# TestGenerateTasks
# ===========================================================================

class TestGenerateTasks:
    """Tests for generate_tasks() mixing logic."""

    @pytest.fixture(autouse=True)
    def import_generate(self):
        import importlib.util
        script_path = os.path.join(SCRIPTS_DIR, "generate_tui_grpo_tasks.py")
        spec = importlib.util.spec_from_file_location(
            "generate_tui_grpo_tasks", script_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.generate_tasks = mod.generate_tasks

    def _make_harmony(self, msg: str, task_n: int = 0) -> str:
        return (
            f"<|developer|>\nYou are an agent.\n"
            f"<|user|>\n{msg}\n"
            f"<|assistant|>\nSure.\n"
            f"<|endoftext|>"
        )

    def test_output_length_respects_target(self):
        """Output should not exceed total_target."""
        tool_texts = [self._make_harmony(f"Tool task {i}") for i in range(10)]
        traj_texts = [self._make_harmony(f"Traj task {i}") for i in range(10)]
        tasks = self.generate_tasks(tool_texts, traj_texts, total_target=8)
        assert len(tasks) <= 8, f"Expected <= 8 tasks, got {len(tasks)}"

    def test_default_mix_approximately_60_40(self):
        """60% tool_calling, 40% agent_traj (±1 due to integer rounding)."""
        tool_texts = [self._make_harmony(f"Tool task {i}") for i in range(100)]
        traj_texts = [self._make_harmony(f"Traj task {i}") for i in range(100)]
        tasks = self.generate_tasks(tool_texts, traj_texts, total_target=10)
        tool_count = sum(1 for t in tasks if t["task_type"] == "tool_calling")
        traj_count = sum(1 for t in tasks if t["task_type"] == "agent_traj")
        # Allow ±1 rounding
        assert 5 <= tool_count <= 7, f"Expected ~60% tool tasks, got {tool_count}/10"
        assert 3 <= traj_count <= 5, f"Expected ~40% traj tasks, got {traj_count}/10"

    def test_each_task_has_description_and_type(self):
        """All output tasks must have 'description' and 'task_type' keys."""
        tool_texts = [self._make_harmony("Do the thing")]
        traj_texts = [self._make_harmony("Fix the bug")]
        tasks = self.generate_tasks(tool_texts, traj_texts, total_target=2)
        for task in tasks:
            assert "description" in task, f"Task missing 'description': {task}"
            assert "task_type" in task, f"Task missing 'task_type': {task}"
            assert task["task_type"] in {"tool_calling", "agent_traj"}

    def test_handles_empty_sources_gracefully(self):
        """Should return an empty list when both sources are empty."""
        tasks = self.generate_tasks([], [], total_target=10)
        assert tasks == [], f"Expected empty list, got {tasks}"

    def test_handles_one_empty_source(self):
        """Should fill from the available source when one is empty."""
        tool_texts = [self._make_harmony(f"Tool task {i}") for i in range(10)]
        tasks = self.generate_tasks(tool_texts, [], total_target=5)
        assert len(tasks) <= 5
        for task in tasks:
            assert task["task_type"] == "tool_calling"
