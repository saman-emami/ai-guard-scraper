import asyncio
import logging
from typing import List, AsyncGenerator, Iterable
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from .ai_models import ModelManager
from .data_model import DetectionResult
from .scraper import AsyncWebScraper

logger = logging.getLogger(__name__)


class AIGuardScraper:
    """
    AI content detection across web pages.

    Combines web scraping with AI model inference, optimizing for either
    CPU or GPU execution environments with appropriate concurrency controls.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        max_scrape_tasks: int = 10,
        cpu_workers: int | None = None,
    ) -> None:

        self.mm = model_manager
        self.cpu_bound = not self.mm.device.startswith("cuda")

        # Use process pool for CPU inference to avoid GIL limitations
        self.pool = (
            ProcessPoolExecutor(max_workers=cpu_workers or max(mp.cpu_count() // 2, 1))
            if self.cpu_bound
            else None
        )

        self.max_scrape_tasks = max_scrape_tasks

    async def _analyse(
        self, url: str, text: str, image_urls: List[str]
    ) -> DetectionResult:
        """
        Run AI detection on extracted content.

        Uses process pool for CPU-bound inference or direct calls for GPU inference.
        """
        start_time = time.perf_counter()

        if self.cpu_bound:
            loop = asyncio.get_running_loop()

            text_prob = await loop.run_in_executor(
                self.pool, self.mm.detect_ai_text, text
            )

            img_probs = await loop.run_in_executor(
                self.pool, self.mm.detect_ai_images, image_urls
            )

        else:
            text_prob = self.mm.detect_ai_text(text)
            img_probs = self.mm.detect_ai_images(image_urls)

        return DetectionResult(
            url=url,
            text=text_prob,
            images=img_probs,
            processing_time=round(time.perf_counter() - start_time, 3),
        )

    async def _process_single_page(
        self, scraper: AsyncWebScraper, url: str
    ) -> DetectionResult:
        """Process a single webpage through the complete scraping and analysis pipeline."""
        try:
            text, image_urls = await scraper.scrape_page(url)
            return await self._analyse(url, text, image_urls)

        except Exception as e:
            return DetectionResult(
                url=url,
                text=None,
                images=None,
                processing_time=0.0,
                error=str(e),
            )

    async def audit(self, urls: Iterable[str]) -> AsyncGenerator[DetectionResult, None]:
        """
        Audit multiple URLs for AI-generated content.

        Yields results as they complete, allowing for streaming processing
        of large URL lists without waiting for all to finish.

        Args:
            urls: Iterable of URLs to analyze

        Yields:
            DetectionResult objects as analysis completes
        """

        semaphore = asyncio.Semaphore(self.max_scrape_tasks)

        async with AsyncWebScraper(self.max_scrape_tasks) as scraper:

            async def _bounded_task(u: str):
                async with semaphore:
                    return await self._process_single_page(scraper, u)

            tasks = {asyncio.create_task(_bounded_task(u)): u for u in urls}

            for task in asyncio.as_completed(tasks):
                yield await task

        if self.pool:
            self.pool.shutdown(wait=False)
