from __future__ import annotations

import aiohttp
import requests
from attr import define, field
from griptape.artifacts import ErrorArtifact, ListArtifact, TextArtifact
from griptape.tools import BaseTool
from griptape.utils.decorators import activity
from schema import Literal, Schema


@define
class WebSearch(BaseTool):
    results_count: int = field(default=5, kw_only=True)
    serper_api_key: str = field(kw_only=True)

    @activity(
        config={
            "description": "Can be used for searching the web",
            "schema": Schema(
                {
                    Literal(
                        "query",
                        description="Search engine request that returns a list of pages with titles, descriptions, and URLs",
                    ): str
                }
            ),
        }
    )
    def search(self, props: dict) -> ListArtifact | ErrorArtifact:
        query = props["values"]["query"]

        try:
            return ListArtifact(
                [TextArtifact(str(result)) for result in self._search_google(query)]
            )
        except Exception as e:
            return ErrorArtifact(f"error searching Google: {e}")

    def _search_google(self, query: str) -> list[dict]:
        url = "https://google.serper.dev/search"

        payload = {"q": query, "num": self.results_count}
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }

        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()

            results = data['organic']
            results = [
                str(
                    {
                        "title": x["title"],
                        "link": x["link"],
                        "description": x["snippet"]
                    }
                )
                for x in results
            ]

            return results

        else:
            raise Exception(
                f"Google Search API returned an error with status code "
                f"{response.status_code} and reason '{response.reason}'"
            )
