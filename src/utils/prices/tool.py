from __future__ import annotations

import aiohttp
import requests
from attr import define, field
from griptape.artifacts import ErrorArtifact, ListArtifact, TextArtifact
from griptape.tools import BaseTool
from griptape.utils.decorators import activity
from schema import Literal, Schema


@define
class Price(BaseTool):
    api_key: str = field(kw_only=True)

    @activity(
        config={
            "description": "Can be used for querying current cryptocurrency prices in USD",
            "schema": Schema(
                {
                    Literal(
                        "ticker",
                        description="Query for the current cryptocurrency asset price by its ticker in USD",
                    ): str
                }
            ),
        }
    )
    def ticker(self, props: dict) -> ListArtifact | ErrorArtifact:
        query = props["values"]["ticker"]

        try:
            return ListArtifact(
                [
                    TextArtifact(f"Current price of {query} is {self._get_price(query)} USD")
                ]
            )
        except Exception as e:
            return ErrorArtifact(f"error getting price: {e}")

    def _get_price(self, query: str) -> list[dict]:
        resp = requests.get(
            "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest",
            params={"symbol": query},
            headers={"X-CMC_PRO_API_KEY": self.api_key},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        return data['data'][query][0]['quote']['USD']['price']
