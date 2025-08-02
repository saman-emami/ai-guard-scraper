import torch
from transformers.pipelines import pipeline
from typing import List, Optional, Dict
import logging
from .data_model import TextAnalysis, ImageAnalysis

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages AI detection models for text and images.

    Uses pre-trained transformers models for classification tasks.
    """

    def __init__(
        self,
        device: str = "auto",
        ai_text_detector_model: str = "fakespot-ai/roberta-base-ai-text-detection-v1",
        ai_image_detector_model: str = "haywoodsloan/ai-image-detector-deploy",
    ):
        self.device = self._get_optimal_device(device)
        self.models_dtype = (
            torch.float16 if self.device.startswith("cuda") else torch.float32
        )

        self.text_pipeline = self._initialize_text_pipeline(ai_text_detector_model)
        self.image_pipeline = self._initialize_image_pipeline(ai_image_detector_model)

    def _get_optimal_device(self, device: str) -> str:
        """Select best available device for model inference."""

        if device == "auto":
            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.current_device()}"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def _initialize_text_pipeline(self, model_name: str):
        """Initialize text classification pipeline."""

        text_classification_pipeline = pipeline(
            "text-classification",
            model=model_name,
            device=self.device,
            torch_dtype=self.models_dtype,
            truncation=True,
            max_length=512,
        )

        logger.info(f"Text classification model loaded successfully on {self.device}")

        return text_classification_pipeline

    def _initialize_image_pipeline(self, model_name: str):
        """Initialize image classification pipeline."""

        image_classification_pipeline = pipeline(
            "image-classification",
            model=model_name,
            device=self.device,
            torch_dtype=self.models_dtype,
        )

        logger.info(f"Image classification model loaded successfully on {self.device}")

        return image_classification_pipeline

    def _get_most_likely_class(self, result: List[Dict]):
        """Extract highest confidence prediction from model output."""
        sorted_result = sorted(result, key=lambda cls: cls["score"], reverse=True)
        most_likely_class = sorted_result[0]

        return most_likely_class

    def _get_ai_likelihood(self, result: List[Dict], ai_class_label: str) -> float:
        """
        Calculate probability that content is AI-generated.

        Returns confidence score for AI class, or (1 - confidence) if
        model predicts non-AI class with high confidence.
        """
        most_likely_class = self._get_most_likely_class(result)

        ai_likelihood = (
            most_likely_class["score"]
            if ai_class_label == most_likely_class["label"]
            else 1 - most_likely_class["score"]
        )

        return ai_likelihood

    def _get_analysis_label(self, result: List[Dict], ai_class_label: str):
        """Convert model prediction to standardized ai/not_ai label."""
        most_likely_class = self._get_most_likely_class(result)

        label = "ai" if ai_class_label == most_likely_class["label"] else "not_ai"

        return label

    def detect_ai_text(
        self, readable_content: str, ai_class_label: str = "AI"
    ) -> Optional[TextAnalysis]:
        """
        Analyze text for AI generation patterns.

        Args:
            readable_content: Input text to analyze
            ai_class_label: Expected label for AI-generated content in model output

        Returns:
            TextAnalysis object or None if analysis fails
        """

        if len(readable_content.strip()) == 0:
            return None

        try:
            result = self.text_pipeline(readable_content)

        except Exception as e:
            logger.warning(f"Text detection failed: {e}")
            return None

        ai_likelihood = self._get_ai_likelihood(result, ai_class_label)
        label = self._get_analysis_label(result, ai_class_label)
        analysis = TextAnalysis(
            label=label, readable_content=readable_content, ai_likelihood=ai_likelihood
        )

        return analysis

    def detect_ai_images(
        self, image_urls: List[str], ai_class_label: str = "artificial"
    ) -> Optional[List[ImageAnalysis]]:
        """
        Analyze images for AI generation patterns using batch processing.

        Args:
            image_urls: List of image URLs to analyze
            ai_class_label: Expected label for AI-generated content in model output

        Returns:
            List of `ImageAnalysis` objects or `None` if analysis fails
        """
        if len(image_urls) == 0:
            return None

        batch_size = 4 if self.device.startswith("cuda") else 1

        model_outputs = []

        try:
            for i in range(0, len(image_urls), batch_size):
                batch = image_urls[i : i + batch_size]
                model_outputs.extend(self.image_pipeline(batch))

        except Exception as e:
            logger.warning(f"Image detection failed: {e}")
            return None

        results = []
        for i, result in enumerate(model_outputs):
            ai_likelihood = self._get_ai_likelihood(result, ai_class_label)
            label = self._get_analysis_label(result, ai_class_label)

            prob_dict = ImageAnalysis(
                label=label, url=image_urls[i], ai_likelihood=ai_likelihood
            )

            results.append(prob_dict)

        return results
