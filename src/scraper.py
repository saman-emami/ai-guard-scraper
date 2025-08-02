import asyncio
import logging
from readability import Document
from bs4 import BeautifulSoup
import re
import playwright.async_api as pw
from typing import List, Tuple
from playwright.async_api import Page
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class AsyncWebScraper:
    """
    Asynchronous web scraper using Playwright for content extraction.

    Extracts readable text content and image URLs from webpages with
    concurrent page processing.
    """

    def __init__(self, max_concurrent_pages: int = 10):
        self.max_concurrent_pages = max_concurrent_pages
        self._browser = None
        self._context = None
        self.semaphore = asyncio.Semaphore(max_concurrent_pages)

    async def __aenter__(self):
        playwright = await pw.async_playwright().start()

        self._browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
            ],
        )

        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (compatible; AIDetectionBot/1.0)",
            java_script_enabled=True,
            ignore_https_errors=True,
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()

    @property
    def context(self):
        """Browser context accessor with initialization check."""
        if self._context is None:
            raise ValueError(
                "Browser not initialized. Use WebScraper within an async context manager."
            )
        return self._context

    async def scrape_page(self, url: str) -> Tuple[str, List[str]]:
        """
        Extract text content and image URLs from a webpage.

        Returns:
            Tuple of (cleaned_text_content, list_of_image_urls)
        """
        async with self.semaphore:
            try:
                page = await self.context.new_page()
            except Exception as e:
                logger.warning(f"Failed to load page at {url}: {e}")
                return "", []

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)

                text_content = await self._extract_text_content(page)

                image_urls = await self._extract_image_urls(page)

                await page.close()
                return text_content, image_urls

            except Exception as e:
                logger.warning(f"Failed to scrape {url}: {e}")
                if "page" in locals():
                    await page.close()
                return "", []

    async def _extract_text_content(self, page: Page) -> str:
        """Extract and clean main text content using readability algorithm."""
        html_content = await page.content()

        doc = Document(html_content)

        title = doc.title()
        summary = doc.summary()

        soup = BeautifulSoup(summary, "html.parser")
        clean_text = soup.get_text(separator=" ", strip=True)

        full_text = f"{title}. {clean_text}" if title else clean_text

        return re.sub(r"\s+", " ", full_text).strip()

    async def _extract_image_urls(
        self, page: Page, min_size: int = 100, image_per_page: int = 10
    ) -> List[str]:
        """
        Extract image URLs that meet size requirements for meaningful analysis.

        Args:
            min_size: Minimum width/height in pixels to filter out icons/decorative images
            image_per_page: Maximum number of valid images per page

        Returns:
            List of valid image URLs
        """

        get_image_data_js_function = """
            () => {
                const images = Array.from(document.querySelectorAll('img'));
                return images.map(img => ({
                    src: img.src,
                    width: img.width,
                    height: img.height
                }));
            }
            """

        image_data = await page.evaluate(get_image_data_js_function)

        valid_image_urls = []

        for img in image_data:

            full_url = urljoin(page.url, img["src"])

            if img["width"] > min_size and img["height"] > min_size:
                valid_image_urls.append(full_url)

            if len(valid_image_urls) >= image_per_page:
                break

        return valid_image_urls
