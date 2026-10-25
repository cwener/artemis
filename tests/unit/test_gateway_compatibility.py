# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider-independent behaviour added on top of upstream.

Two paths differ from upstream because they only make sense for non-Gemini
deployments; both must leave the native (Gemini) behaviour untouched:

* ``nodes`` entries for the optional judge nodes must be honoured instead of
  being dropped, while unconfigured judges keep the cheap flash-lite default.
* the checker requests its verdict as plain JSON on OpenAI-compatible gateways
  (which support forced-tool-call structured output only partially) and keeps
  ``with_structured_output`` for native providers.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from artemis.agents.checker import checker
from artemis.agents.checker.checker import (
    CheckReport,
    _is_gateway_llm,
    _structured_report,
)
from artemis.config.llm import LLMConfig, _expand_default_into_nodes


def _unified_config(nodes: dict) -> dict:
    return {
        "default": {
            "provider": "openai",
            "model": "main-model",
            "fallback": {"provider": "openai", "model": "fallback-model"},
        },
        "nodes": nodes,
    }


def test_unconfigured_optional_judge_nodes_keep_the_lightweight_default():
    llm_cfg = LLMConfig.model_validate(_expand_default_into_nodes(_unified_config({})))

    node = llm_cfg.get_agent("planner_validation")
    safety_net = llm_cfg.get_agent("validator_pixel_safety_net")

    assert node.provider == "google" and "lite" in node.model
    assert safety_net.provider == "google" and "lite" in safety_net.model


def test_configured_optional_judge_node_override_is_honoured():
    llm_cfg = LLMConfig.model_validate(
        _expand_default_into_nodes(
            _unified_config({"planner_validation": {"model": "judge-model"}})
        )
    )

    node = llm_cfg.get_agent("planner_validation")

    assert node.model == "judge-model"
    # `provider` is inherited from the `default` block.
    assert node.provider == "openai"
    # The sibling stays on the lightweight default.
    assert "lite" in llm_cfg.get_agent("validator_pixel_safety_net").model


@pytest.mark.asyncio
async def test_gateway_checker_requests_plain_json(monkeypatch):
    report = CheckReport(verdicts=[])
    seen: dict = {}

    async def fake_acomplete_structured(llm, messages, *, schema=None):
        seen["schema"] = schema
        seen["messages"] = messages
        return report

    monkeypatch.setattr(checker, "acomplete_structured", fake_acomplete_structured)
    gateway_llm = SimpleNamespace(endpoint=SimpleNamespace(provider="openai"))

    assert _is_gateway_llm(gateway_llm)
    result = await _structured_report(gateway_llm, [HumanMessage(content="go")])

    assert result is report
    assert seen["schema"] is CheckReport
    assert "JSON schema" in seen["messages"][-1].content


@pytest.mark.asyncio
async def test_native_checker_keeps_structured_output_path():
    class NativeLLM:
        endpoint = SimpleNamespace(provider="google")

        def __init__(self):
            self.calls = 0

        def with_structured_output(self, schema):
            self.calls += 1

            class _Bound:
                async def ainvoke(self, messages):
                    return CheckReport(verdicts=[])

            return _Bound()

    llm = NativeLLM()

    assert not _is_gateway_llm(llm)
    result = await _structured_report(llm, [HumanMessage(content="go")])

    assert isinstance(result, CheckReport)
    assert llm.calls == 1
