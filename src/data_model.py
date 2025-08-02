from dataclasses import dataclass
from typing import Literal, Optional, List


@dataclass
class TextAnalysis:
    label: Literal["ai", "not_ai"]
    readable_content: str
    ai_likelihood: float


@dataclass
class ImageAnalysis:
    label: Literal["ai", "not_ai"]
    url: str
    ai_likelihood: float


@dataclass
class DetectionResult:
    url: str
    text: Optional[TextAnalysis]
    images: Optional[List[ImageAnalysis]]
    processing_time: float
    error: Optional[str] = None
