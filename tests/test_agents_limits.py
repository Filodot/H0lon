"""Subscription window utilisation from Claude's rate_limit_event."""

from __future__ import annotations

import json

from h0lon.agents.base import format_limits
from h0lon.agents.claude import ClaudeStream, parse_rate_limit_info

# Shape observed from Claude Code 2.1.288 (2026-10-03).
EVENT = {
    "type": "rate_limit_event",
    "rate_limit_info": {
        "status": "allowed",
        "resetsAt": 1791048000,
        "rateLimitType": "five_hour",
        "overageStatus": "rejected",
        "overageDisabledReason": "org_level_disabled",
        "isUsingOverage": False,
        "unifiedWindows": {
            "five_hour": {"utilization": 0.49, "resetsAt": 1791048000},
            "seven_day": {"utilization": 0.22, "resetsAt": 1791259200},
        },
    },
    "uuid": "u",
    "session_id": "s",
}


def test_parse_rate_limit_info_normalises_windows() -> None:
    limits = parse_rate_limit_info(EVENT["rate_limit_info"])
    assert limits is not None
    assert limits["status"] == "allowed"
    assert limits["type"] == "five_hour"
    assert limits["windows"]["five_hour"]["utilization"] == 0.49
    assert limits["windows"]["seven_day"]["resets_at"].startswith("2026-")
    assert limits["using_overage"] is False


def test_parse_rate_limit_info_tolerates_garbage() -> None:
    assert parse_rate_limit_info(None) is None
    assert parse_rate_limit_info("x") is None
    limits = parse_rate_limit_info({"unifiedWindows": {"five_hour": "bad", "x": {}}})
    assert limits is not None
    assert limits["windows"] == {"x": {"utilization": None, "resets_at": None}}


def test_stream_keeps_limits_and_is_quiet_when_low() -> None:
    stream = ClaudeStream()
    assert stream.feed(json.dumps(EVENT)) is None
    assert stream.limits is not None
    assert stream.limits["windows"]["five_hour"]["utilization"] == 0.49


def test_stream_reports_hot_or_rejected_window() -> None:
    hot = json.loads(json.dumps(EVENT))
    hot["rate_limit_info"]["unifiedWindows"]["five_hour"]["utilization"] = 0.93
    msg = ClaudeStream().feed(json.dumps(hot))
    assert msg is not None and "five_hour 93 %" in msg
    rejected = json.loads(json.dumps(EVENT))
    rejected["rate_limit_info"]["status"] = "rejected"
    msg = ClaudeStream().feed(json.dumps(rejected))
    assert msg is not None and "rejected" in msg


def test_format_limits() -> None:
    text = format_limits(parse_rate_limit_info(EVENT["rate_limit_info"]))
    assert text is not None
    assert text.startswith("5 ч — 49 % (сброс ")
    assert "7 дн — 22 %" in text
    assert format_limits(None) is None
    assert format_limits({"windows": {}}) is None


def _proc(**kw: object) -> object:
    from h0lon.procutil import ProcResult

    defaults: dict[str, object] = {
        "argv": ["claude"],
        "exit_code": 1,
        "stdout": "",
        "stderr": "",
        "duration_s": 0.1,
        "stopped": True,
    }
    defaults.update(kw)
    return ProcResult(**defaults)  # type: ignore[arg-type]


def test_auth_retry_marks_stream_fatal_and_classifies_auth() -> None:
    from h0lon.agents.claude import classify

    stream = ClaudeStream()
    stream.feed(
        json.dumps(
            {
                "type": "system",
                "subtype": "api_retry",
                "attempt": 1,
                "max_retries": 10,
                "error": "authentication_failed",
                "error_status": 401,
            }
        )
    )
    assert stream.fatal == "authentication_failed"
    kind, _ = classify(stream, _proc(), max_turns=5)  # type: ignore[arg-type]
    assert kind == "auth"


def test_rejected_window_marks_fatal_and_classifies_rate_limit() -> None:
    from h0lon.agents.claude import classify

    rejected = json.loads(json.dumps(EVENT))
    rejected["rate_limit_info"]["status"] = "rejected"
    stream = ClaudeStream()
    stream.feed(json.dumps(rejected))
    assert stream.fatal == "rate_limit"
    kind, message = classify(stream, _proc(), max_turns=5)  # type: ignore[arg-type]
    assert kind == "rate_limit"
    assert message and "rejected" in message


def test_transient_retry_is_not_fatal() -> None:
    stream = ClaudeStream()
    stream.feed(
        json.dumps({"type": "system", "subtype": "api_retry", "attempt": 1, "error": "overloaded"})
    )
    assert stream.fatal is None
