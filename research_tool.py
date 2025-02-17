"""
title: Research Tool
author: constLiakos with enhancements by justinh-rahb and ther3zz, iamarcel
funding_url: https://github.com/open-webui
version: 0.1.0
license: MIT
"""

import os
import requests
from datetime import datetime
import json
from requests import get
import concurrent.futures
from urllib.parse import urlparse, urljoin
import re
import unicodedata
from pydantic import BaseModel, Field
import asyncio
from typing import Callable, Any, Optional


class HelpFunctions:
    def __init__(self):
        pass

    def get_base_url(self, url):
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return base_url

    def generate_excerpt(self, content, max_length=200):
        return content[:max_length] + "..." if len(content) > max_length else content

    def remove_emojis(self, text):
        return "".join(c for c in text if not unicodedata.category(c).startswith("So"))

    def process_search_result(self, result, valves):
        title_site = self.remove_emojis(result["title"])
        url_site = result["url"]
        snippet = result.get("content", "")

        # Check if the website is in the ignored list, but only if IGNORED_WEBSITES is not empty
        if valves.IGNORED_WEBSITES:
            base_url = self.get_base_url(url_site)
            if any(
                ignored_site.strip() in base_url
                for ignored_site in valves.IGNORED_WEBSITES.split(",")
            ):
                return None

        try:
            content_site = self.web_scrape(url_site)

            truncated_content = self.truncate_to_n_words(
                content_site, valves.PAGE_CONTENT_WORDS_LIMIT
            )

            return {
                "title": title_site,
                "url": url_site,
                "content": truncated_content,
                "snippet": self.remove_emojis(snippet),
            }

        except requests.exceptions.RequestException as e:
            return None

    def truncate_to_n_words(self, text, token_limit):
        tokens = text.split()
        truncated_tokens = tokens[:token_limit]
        return " ".join(truncated_tokens)

    def web_scrape(self, url: str) -> str:
        """
        Scrape and process a web page using r.jina.ai

        :param url: The URL of the web page to scrape.
        :return: The scraped and processed content without the Links/Buttons section, or an error message.
        """
        jina_url = f"https://r.jina.ai/{url}"

        headers = {
            "X-No-Cache": "true",
            "X-With-Images-Summary": "true",
            "X-With-Links-Summary": "true",
        }

        try:
            response = requests.get(jina_url, headers=headers)
            response.raise_for_status()

            # Extract content and remove Links/Buttons section as its too many tokens
            content = response.text
            links_section_start = content.rfind("Images:")
            if links_section_start != -1:
                content = content[:links_section_start].strip()

            return content

        except requests.RequestException as e:
            raise Exception(f"Error scraping web page: {str(e)}")


class EventEmitter:
    def __init__(self, event_emitter: Optional[Callable[[dict], Any]] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        SEARXNG_ENGINE_API_BASE_URL: str = Field(
            default="https://example.com/search",
            description="The base URL for Search Engine",
        )
        IGNORED_WEBSITES: str = Field(
            default="",
            description="Comma-separated list of websites to ignore",
        )
        RETURNED_SCRAPPED_PAGES_NO: int = Field(
            default=3,
            description="The number of Search Engine Results to Parse",
        )
        SCRAPPED_PAGES_NO: int = Field(
            default=5,
            description="Total pages scapped. Ideally greater than one of the returned pages",
        )
        PAGE_CONTENT_WORDS_LIMIT: int = Field(
            default=5000,
            description="Limit words content for each page.",
        )
        CITATION_LINKS: bool = Field(
            default=False,
            description="If True, send custom citations with links",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

    async def search_web(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search the web and get the content of the relevant pages. Search for unknown knowledge, news, info, public contact info, weather, etc.
        :params query: Web Query used in search engine.
        :return: The content of the pages in json format.
        """
        functions = HelpFunctions()
        emitter = EventEmitter(__event_emitter__)

        await emitter.emit(f"Initiating web search for: {query}")

        search_engine_url = self.valves.SEARXNG_ENGINE_API_BASE_URL

        # Ensure RETURNED_SCRAPPED_PAGES_NO does not exceed SCRAPPED_PAGES_NO
        if self.valves.RETURNED_SCRAPPED_PAGES_NO > self.valves.SCRAPPED_PAGES_NO:
            self.valves.RETURNED_SCRAPPED_PAGES_NO = self.valves.SCRAPPED_PAGES_NO

        params = {
            "q": query,
            "format": "json",
            "number_of_results": self.valves.RETURNED_SCRAPPED_PAGES_NO,
        }

        try:
            await emitter.emit("Sending request to search engine")
            resp = requests.get(
                search_engine_url, params=params, headers=self.headers, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            limited_results = results[: self.valves.SCRAPPED_PAGES_NO]
            await emitter.emit(f"Retrieved {len(limited_results)} search results")

        except requests.exceptions.RequestException as e:
            await emitter.emit(
                status="error",
                description=f"Error during search: {str(e)}",
                done=True,
            )
            return json.dumps({"error": str(e)})

        results_json = []
        if limited_results:
            await emitter.emit(f"Processing search results")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        functions.process_search_result, result, self.valves
                    )
                    for result in limited_results
                ]
                for future in concurrent.futures.as_completed(futures):
                    result_json = future.result()
                    if result_json:
                        try:
                            json.dumps(result_json)
                            results_json.append(result_json)
                        except (TypeError, ValueError):
                            continue
                    if len(results_json) >= self.valves.RETURNED_SCRAPPED_PAGES_NO:
                        break

            results_json = results_json[: self.valves.RETURNED_SCRAPPED_PAGES_NO]

            if self.valves.CITATION_LINKS and __event_emitter__:
                for result in results_json:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [result["content"]],
                                "metadata": [{"source": result["url"]}],
                                "source": {"name": result["title"]},
                            },
                        }
                    )

        await emitter.emit(
            status="complete",
            description=f"Web search completed. Retrieved content from {len(results_json)} pages",
            done=True,
        )

        return json.dumps(results_json, ensure_ascii=False)
