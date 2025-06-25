# nano-vllm Coding Instructions

## Project Overview

We are working with a PyTorch-based LLM inference engine (nano-vllm) focused on efficient text generation for Qwen3 models with optimized attention and caching mechanisms.

---

## General Coding Guidelines

- **Type Hints:** Always include proper type hints for function parameters and return values in Python code to maintain clarity and compatibility.
- **Tensor Shape Comments:** Use tensor shape comments for all tensor operations in PyTorch. Include comments like `# [batch_size, seq_len, hidden_dim]` before operating on tensors.
- **Block Allocation in KV Cache:** When dealing with block allocation in KV cache, be aware of the relationship between sequence length and block_size. Avoid strict assertions and handle edge cases gracefully.
- **Debugging:** Prefer adding explicit validation steps and informative error messages rather than using assertions that might fail silently.
- **Tensor Device Checks:** Always check tensor devices before operations to ensure compatibility between CPU, CUDA, and MPS (Apple Silicon).
- **Grouped-Query Attention:** When implementing grouped-query attention, verify shapes after reshaping operations and ensure proper handling of causal masks.
- **Sequence Position Tracking:** When tracking sequence positions, ensure the position property in `sequence.py` is correctly used and position values match KV cache indices.

---

## Testing and Benchmarking

- **Block Manager Logic:** Use `test_block_allocation.py` to verify KV cache block manager logic.
- **Model Integration Testing:** Use `test_nano_vllm.py` for model integration testing.
- **Benchmarking Pattern:** Our benchmarking follows a specific pattern: start with small batches in `mini_bench.py` before scaling to full tests in `bench.py`.

---

## Sampling Logic

- **Temperature Scaling:** Ensure temperature scaling is correct.
- **Repetition Penalty:** Repetition penalty must be properly applied across batch dimensions.

---

## Documentation

- **Algorithm Documentation:** Document all complex algorithms and non-obvious optimizations, especially in the block manager and attention implementation.

---

## Formatting and Style Guide

- **Imports:** Keep imports organized and grouped by standard library, third-party libraries, and local imports at the top of the file. Remove unused imports.
- **Indentation:** Use 4 spaces for indentation.
- **Strings:** Use double quotes for strings.
- **String Formatting:** Use f-strings for string formatting.
- **Type Hints:** Use type hints for function parameters and return values.
- **Naming Conventions:** Use snake_case for function and variable names and _ for private variables. Use CamelCase for class names.
- **Docstrings:** Use docstrings for public methods and functions.

---

## Testing and Debugging Practices

- **Testing Framework:** Use pytest for testing.
- **Mocking:** Mock external dependencies where necessary.
- **Unit Tests:** Write unit tests for all public methods and functions.
- **Fixtures:** Use fixtures for setup and teardown of test environments.
- **Assertions:** Use assert statements for testing expected outcomes.
- **Dev Dependencies:** Always add test and coverage tools (e.g., pytest, pytest-cov) to the `[dependency-groups.dev]` section of `pyproject.toml` and install them in CI using `uv sync --dev`. Do not install them manually in the workflow.
- **Logging:** Use logging for debugging and error messages.
- **Parameterized Tests:** Use `pytest.mark.parametrize` for parameterized tests.
- **Test Isolation:** Ensure tests are isolated and do not depend on each other.
- **Test Data:** Setup tests to run on a smaller subset of data before scaling up.

---

## Standard Libraries and Dependencies

- **Standard Libraries:** Use standard libraries like `os`, `sys`, `json`, `logging`, and `itertools`.
- **Third-Party Libraries:** Use third-party libraries like `torch`, `numpy`, `pytest`, `requests`/`httpx`, `fastapi`, and `pydantic`.
- **Deprecated Libraries:** Avoid using deprecated or outdated libraries.
- **Dependency Management:** Manage dependencies using `pyproject.toml` and uv package manager.
- **Formatting and Type Checking:** Format files with ruff and check for type hints