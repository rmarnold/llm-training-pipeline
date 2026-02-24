"""Tests for the evaluator dispatch registry system.

Covers:
- EVALUATORS registry population (rust, python)
- get_evaluator() lookup and ValueError for unknown languages
- Dynamic registration and cleanup via register_evaluator decorator
- compute_execution_reward and rank_solutions_by_execution dispatch routing
- Backward-compatibility: rust_evaluators module-level function exports
"""
import inspect
import pytest
from unittest.mock import MagicMock, patch

from pipeline_lib.evaluator_dispatch import (
    EVALUATORS,
    compute_execution_reward,
    get_evaluator,
    rank_solutions_by_execution,
    register_evaluator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MinimalEvaluator:
    """Minimal evaluator stub used to exercise registry mechanics."""

    def compute_execution_reward(
        self,
        code: str,
        tool_calls=None,
        reward_config=None,
    ) -> float:
        return 0.0

    def rank_solutions_by_execution(
        self,
        solutions,
        tests_code=None,
    ):
        return [(s, 0.0, {}) for s in solutions]


# ---------------------------------------------------------------------------
# TestEvaluatorRegistry
# ---------------------------------------------------------------------------

class TestEvaluatorRegistry:
    """Test the EVALUATORS dict and get_evaluator() factory."""

    def test_rust_evaluator_registered(self):
        """'rust' key must exist in EVALUATORS after module import."""
        assert "rust" in EVALUATORS, (
            "RustEvaluator was not registered. "
            "Check that rust_evaluators.py is imported and uses @register_evaluator('rust')."
        )

    def test_python_evaluator_registered(self):
        """'python' key must exist in EVALUATORS after module import."""
        assert "python" in EVALUATORS, (
            "PythonEvaluator was not registered. "
            "Check that python_evaluators.py is imported and uses @register_evaluator('python')."
        )

    def test_get_evaluator_rust_returns_instance_with_required_methods(self):
        """get_evaluator('rust') must return an object implementing the protocol."""
        evaluator = get_evaluator("rust")
        assert hasattr(evaluator, "compute_execution_reward"), (
            "RustEvaluator instance missing compute_execution_reward method"
        )
        assert hasattr(evaluator, "rank_solutions_by_execution"), (
            "RustEvaluator instance missing rank_solutions_by_execution method"
        )
        assert callable(evaluator.compute_execution_reward)
        assert callable(evaluator.rank_solutions_by_execution)

    def test_get_evaluator_python_returns_instance_with_required_methods(self):
        """get_evaluator('python') must return an object implementing the protocol."""
        evaluator = get_evaluator("python")
        assert hasattr(evaluator, "compute_execution_reward"), (
            "PythonEvaluator instance missing compute_execution_reward method"
        )
        assert hasattr(evaluator, "rank_solutions_by_execution"), (
            "PythonEvaluator instance missing rank_solutions_by_execution method"
        )
        assert callable(evaluator.compute_execution_reward)
        assert callable(evaluator.rank_solutions_by_execution)

    def test_get_evaluator_unknown_language_raises_value_error(self):
        """get_evaluator with an unregistered language must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_evaluator("cobol")
        error_msg = str(exc_info.value)
        assert "cobol" in error_msg, (
            "ValueError message should name the unknown language"
        )

    def test_get_evaluator_error_message_lists_available_languages(self):
        """ValueError for unknown language should mention available languages."""
        with pytest.raises(ValueError) as exc_info:
            get_evaluator("cobol")
        error_msg = str(exc_info.value)
        # At minimum, the registered languages should appear in the message
        assert "rust" in error_msg or "python" in error_msg or "Available" in error_msg

    def test_register_custom_evaluator_and_retrieve(self):
        """register_evaluator decorator must add a class to EVALUATORS."""
        lang = "test_lang_custom"
        assert lang not in EVALUATORS, "Precondition: test_lang_custom must not exist before test"

        try:
            @register_evaluator(lang)
            class _CustomEvaluator(_MinimalEvaluator):
                pass

            assert lang in EVALUATORS, "Custom evaluator was not added to EVALUATORS"
            instance = get_evaluator(lang)
            assert isinstance(instance, _CustomEvaluator)
            assert hasattr(instance, "compute_execution_reward")
            assert hasattr(instance, "rank_solutions_by_execution")
        finally:
            # Teardown: remove the test entry to avoid polluting other tests
            EVALUATORS.pop(lang, None)

    def test_register_evaluator_returns_class_unchanged(self):
        """register_evaluator must return the decorated class itself."""
        lang = "test_lang_return_check"
        try:
            @register_evaluator(lang)
            class _ReturnCheckEvaluator(_MinimalEvaluator):
                pass

            assert EVALUATORS[lang] is _ReturnCheckEvaluator, (
                "register_evaluator must store the exact class in the registry"
            )
        finally:
            EVALUATORS.pop(lang, None)

    def test_get_evaluator_returns_new_instance_each_call(self):
        """Each call to get_evaluator should return a fresh instance."""
        a = get_evaluator("rust")
        b = get_evaluator("rust")
        assert a is not b, (
            "get_evaluator should instantiate a new object on each call"
        )


# ---------------------------------------------------------------------------
# TestDispatchFunctions
# ---------------------------------------------------------------------------

class TestDispatchFunctions:
    """Test that module-level dispatch functions route to the correct evaluator."""

    def test_compute_execution_reward_default_language_is_rust(self):
        """compute_execution_reward without language kwarg must route to RustEvaluator."""
        mock_evaluator = MagicMock()
        mock_evaluator.compute_execution_reward.return_value = 0.5

        with patch(
            "pipeline_lib.evaluator_dispatch.get_evaluator",
            return_value=mock_evaluator,
        ) as mock_get:
            result = compute_execution_reward("fn main() {}")

        mock_get.assert_called_once_with("rust")
        mock_evaluator.compute_execution_reward.assert_called_once_with(
            "fn main() {}",
            tool_calls=None,
            reward_config=None,
        )
        assert result == 0.5

    def test_compute_execution_reward_explicit_rust(self):
        """compute_execution_reward(language='rust') must call get_evaluator('rust')."""
        mock_evaluator = MagicMock()
        mock_evaluator.compute_execution_reward.return_value = 1.0

        with patch(
            "pipeline_lib.evaluator_dispatch.get_evaluator",
            return_value=mock_evaluator,
        ) as mock_get:
            result = compute_execution_reward("fn main() {}", language="rust")

        mock_get.assert_called_once_with("rust")
        assert result == 1.0

    def test_compute_execution_reward_python_dispatch(self):
        """compute_execution_reward(language='python') must route to PythonEvaluator."""
        mock_evaluator = MagicMock()
        mock_evaluator.compute_execution_reward.return_value = 0.7

        with patch(
            "pipeline_lib.evaluator_dispatch.get_evaluator",
            return_value=mock_evaluator,
        ) as mock_get:
            result = compute_execution_reward(
                "def hello(): pass",
                language="python",
            )

        mock_get.assert_called_once_with("python")
        mock_evaluator.compute_execution_reward.assert_called_once_with(
            "def hello(): pass",
            tool_calls=None,
            reward_config=None,
        )
        assert result == 0.7

    def test_compute_execution_reward_passes_tool_calls_and_reward_config(self):
        """Dispatch function must forward tool_calls and reward_config arguments."""
        mock_evaluator = MagicMock()
        mock_evaluator.compute_execution_reward.return_value = -1.0

        tool_calls = [{"name": "cargo_check", "arguments": "{}"}]
        reward_config = {"compilation_failure": -0.5}

        with patch(
            "pipeline_lib.evaluator_dispatch.get_evaluator",
            return_value=mock_evaluator,
        ):
            compute_execution_reward(
                "fn main() {}",
                language="rust",
                tool_calls=tool_calls,
                reward_config=reward_config,
            )

        mock_evaluator.compute_execution_reward.assert_called_once_with(
            "fn main() {}",
            tool_calls=tool_calls,
            reward_config=reward_config,
        )

    def test_rank_solutions_default_language_is_rust(self):
        """rank_solutions_by_execution without language kwarg must route to RustEvaluator."""
        mock_evaluator = MagicMock()
        mock_evaluator.rank_solutions_by_execution.return_value = [
            ("fn main() {}", 4.0, {"check_passed": True}),
        ]

        with patch(
            "pipeline_lib.evaluator_dispatch.get_evaluator",
            return_value=mock_evaluator,
        ) as mock_get:
            result = rank_solutions_by_execution(["fn main() {}"])

        mock_get.assert_called_once_with("rust")
        mock_evaluator.rank_solutions_by_execution.assert_called_once_with(
            ["fn main() {}"],
            tests_code=None,
        )
        assert result[0][1] == 4.0

    def test_rank_solutions_explicit_rust(self):
        """rank_solutions_by_execution(language='rust') must call get_evaluator('rust')."""
        mock_evaluator = MagicMock()
        mock_evaluator.rank_solutions_by_execution.return_value = []

        with patch(
            "pipeline_lib.evaluator_dispatch.get_evaluator",
            return_value=mock_evaluator,
        ) as mock_get:
            rank_solutions_by_execution(["fn main() {}"], language="rust")

        mock_get.assert_called_once_with("rust")

    def test_rank_solutions_python_dispatch(self):
        """rank_solutions_by_execution(language='python') must route to PythonEvaluator."""
        mock_evaluator = MagicMock()
        mock_evaluator.rank_solutions_by_execution.return_value = [
            ("def hello(): pass", 6.5, {"syntax_valid": True}),
        ]

        with patch(
            "pipeline_lib.evaluator_dispatch.get_evaluator",
            return_value=mock_evaluator,
        ) as mock_get:
            result = rank_solutions_by_execution(
                ["def hello(): pass"],
                language="python",
            )

        mock_get.assert_called_once_with("python")
        mock_evaluator.rank_solutions_by_execution.assert_called_once_with(
            ["def hello(): pass"],
            tests_code=None,
        )
        assert result[0][1] == 6.5

    def test_rank_solutions_passes_tests_code(self):
        """rank_solutions_by_execution must forward tests_code to the evaluator."""
        mock_evaluator = MagicMock()
        mock_evaluator.rank_solutions_by_execution.return_value = []

        tests_code = "#[test] fn it_works() { assert!(true); }"

        with patch(
            "pipeline_lib.evaluator_dispatch.get_evaluator",
            return_value=mock_evaluator,
        ):
            rank_solutions_by_execution(
                ["fn add(a: i32, b: i32) -> i32 { a + b }"],
                language="rust",
                tests_code=tests_code,
            )

        mock_evaluator.rank_solutions_by_execution.assert_called_once_with(
            ["fn add(a: i32, b: i32) -> i32 { a + b }"],
            tests_code=tests_code,
        )


# ---------------------------------------------------------------------------
# TestBackwardCompatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Verify that pre-dispatch rust_evaluators module-level API still works."""

    def test_rust_evaluators_exports_compute_execution_reward(self):
        """rust_evaluators.compute_execution_reward must be importable."""
        from pipeline_lib.rust_evaluators import compute_execution_reward as rust_cer
        assert callable(rust_cer), (
            "compute_execution_reward is not callable in rust_evaluators"
        )

    def test_rust_evaluators_exports_rank_solutions_by_execution(self):
        """rust_evaluators.rank_solutions_by_execution must be importable."""
        from pipeline_lib.rust_evaluators import rank_solutions_by_execution as rust_rank
        assert callable(rust_rank), (
            "rank_solutions_by_execution is not callable in rust_evaluators"
        )

    def test_rust_evaluators_exports_validate_tool_call(self):
        """rust_evaluators.validate_tool_call must be importable."""
        from pipeline_lib.rust_evaluators import validate_tool_call
        assert callable(validate_tool_call), (
            "validate_tool_call is not callable in rust_evaluators"
        )

    def test_dispatch_and_rust_evaluators_compute_reward_share_core_args(self):
        """Both dispatch and rust_evaluators compute_execution_reward accept code/tool_calls/reward_config."""
        from pipeline_lib.rust_evaluators import (
            compute_execution_reward as rust_cer,
        )

        dispatch_params = set(inspect.signature(compute_execution_reward).parameters)
        rust_params = set(inspect.signature(rust_cer).parameters)

        core_args = {"code", "tool_calls", "reward_config"}
        assert core_args.issubset(dispatch_params), (
            f"dispatch compute_execution_reward missing params: {core_args - dispatch_params}"
        )
        assert core_args.issubset(rust_params), (
            f"rust_evaluators compute_execution_reward missing params: {core_args - rust_params}"
        )

    def test_dispatch_and_rust_evaluators_rank_solutions_share_core_args(self):
        """Both dispatch and rust_evaluators rank_solutions_by_execution accept solutions/tests_code."""
        from pipeline_lib.rust_evaluators import (
            rank_solutions_by_execution as rust_rank,
        )

        dispatch_params = set(inspect.signature(rank_solutions_by_execution).parameters)
        rust_params = set(inspect.signature(rust_rank).parameters)

        core_args = {"solutions", "tests_code"}
        assert core_args.issubset(dispatch_params), (
            f"dispatch rank_solutions_by_execution missing params: {core_args - dispatch_params}"
        )
        assert core_args.issubset(rust_params), (
            f"rust_evaluators rank_solutions_by_execution missing params: {core_args - rust_params}"
        )

    def test_dispatch_compute_reward_language_kwarg_defaults_to_rust(self):
        """dispatch compute_execution_reward must default language to 'rust'."""
        sig = inspect.signature(compute_execution_reward)
        language_param = sig.parameters.get("language")
        assert language_param is not None, "dispatch function has no 'language' parameter"
        assert language_param.default == "rust", (
            f"Expected default language='rust', got '{language_param.default}'"
        )

    def test_dispatch_rank_solutions_language_kwarg_defaults_to_rust(self):
        """dispatch rank_solutions_by_execution must default language to 'rust'."""
        sig = inspect.signature(rank_solutions_by_execution)
        language_param = sig.parameters.get("language")
        assert language_param is not None, "dispatch function has no 'language' parameter"
        assert language_param.default == "rust", (
            f"Expected default language='rust', got '{language_param.default}'"
        )

    def test_validate_tool_call_accepts_valid_tool_call(self):
        """validate_tool_call must not require subprocess calls for basic validation."""
        from pipeline_lib.rust_evaluators import validate_tool_call

        valid_tc = {"name": "cargo_check", "arguments": "{}"}
        assert validate_tool_call(valid_tc) is True

    def test_validate_tool_call_rejects_missing_name(self):
        """validate_tool_call must reject dicts with no name field."""
        from pipeline_lib.rust_evaluators import validate_tool_call

        invalid_tc = {"arguments": "{}"}
        assert validate_tool_call(invalid_tc) is False

    def test_validate_tool_call_rejects_malformed_arguments_json(self):
        """validate_tool_call must reject dicts with invalid JSON in arguments."""
        from pipeline_lib.rust_evaluators import validate_tool_call

        invalid_tc = {"name": "cargo_check", "arguments": "{not valid json"}
        assert validate_tool_call(invalid_tc) is False
