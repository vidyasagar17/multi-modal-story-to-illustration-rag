"""Offline tests for the planning loop's behaviour when the API stops answering."""

import json

import httpx
import pytest
from openai import APIStatusError

from storyillus.cli import _plan_pages
from storyillus.models import Page

REPLY = json.dumps(
    {
        "summary": "Victor watches the creature open its eye.",
        "characters": ["Victor Frankenstein"],
        "setting": "a candlelit laboratory",
        "mood": "dread",
        "key_visual": "A pale figure sitting up beneath a sheet.",
    }
)

PAGES = [Page(index=i, text=f"page {i} text") for i in range(1, 6)]


class StoppingLLM:
    """Answers `limit` times, then fails the way a depleted quota does."""

    def __init__(self, limit: int, status: int = 402) -> None:
        self.limit = limit
        self.status = status
        self.calls = 0

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls += 1
        if self.calls > self.limit:
            response = httpx.Response(
                self.status, request=httpx.Request("POST", "https://router.example/v1")
            )
            raise APIStatusError("out of credits", response=response, body=None)
        return REPLY


def test_pages_planned_before_the_failure_are_returned():
    planned, stopped = _plan_pages(StoppingLLM(limit=3), PAGES)

    assert [page["index"] for page in planned] == [1, 2, 3]
    assert stopped is not None
    assert stopped.status_code == 402


def test_a_clean_run_reports_no_failure():
    planned, stopped = _plan_pages(StoppingLLM(limit=len(PAGES)), PAGES)

    assert len(planned) == len(PAGES)
    assert stopped is None


def test_a_failure_on_the_first_page_returns_nothing_and_the_error():
    planned, stopped = _plan_pages(StoppingLLM(limit=0), PAGES)

    assert planned == []
    assert stopped is not None


def test_the_error_is_not_swallowed_into_a_plan():
    """A stopped run must be distinguishable from a run that planned every page."""
    planned, stopped = _plan_pages(StoppingLLM(limit=2), PAGES)

    assert len(planned) < len(PAGES)
    with pytest.raises(APIStatusError):
        raise stopped
