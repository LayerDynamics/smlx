#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Donut Processor.

Handles document image preprocessing and tokenization for Donut model.
"""

from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np
from PIL import Image


class DonutProcessor:
    """
    Processor for Donut model combining image processing and tokenization.

    Args:
        image_size: Target image size (default: (224, 224))
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
    """

    def __init__(
        self,
        tokenizer=None,
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def process_image(self, image: Union[str, Path, Image.Image]) -> mx.array:
        """
        Process document image for model input.

        Args:
            image: Image file path or PIL Image

        Returns:
            Processed image tensor
                Shape: (1, 3, height, width)

        Example:
            >>> processor = DonutProcessor()
            >>> image_tensor = processor.process_image("document.jpg")
            >>> print(image_tensor.shape)  # (1, 3, 224, 224)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = load_image(image)

        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")

        # Resize
        image = image.resize(self.image_size, Image.BICUBIC)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array and normalize
        image_np = np.array(image, dtype=np.float32) / 255.0

        # Normalize with mean and std
        image_np = (image_np - self.mean) / self.std

        # Convert to MLX array
        # Shape: (H, W, 3) -> (3, H, W)
        image_mx = mx.array(image_np.transpose(2, 0, 1))

        # Add batch dimension
        # Shape: (3, H, W) -> (1, 3, H, W)
        image_mx = mx.expand_dims(image_mx, axis=0)

        return image_mx

    def process_document(
        self,
        image: Union[str, Path, Image.Image],
        task_prompt: str = "<s_cord-v2>",
        max_length: int = 512,
    ) -> dict:
        """
        Process document image for structured extraction.

        This prepares inputs for the Donut model to generate structured output
        from document images (receipts, forms, documents, etc.).

        Pipeline:
            1. Load and preprocess image
            2. Encode task-specific prompt (e.g., <s_cord-v2> for receipts)
            3. Return prepared inputs for model inference

        Args:
            image: Document image (path or PIL Image)
            task_prompt: Task specification token
                - "<s_cord-v2>": Receipt parsing (menu items, prices)
                - "<s_docvqa>": Document question answering
                - "<s_rvlcdip>": Document classification
                - Custom task tokens as needed
            max_length: Maximum sequence length for tokenization

        Returns:
            Dict with:
                - pixel_values: Preprocessed image tensor [1, 3, H, W]
                - decoder_input_ids: Task prompt tokens [1, seq_len]
                - task_prompt: Original task string (for reference)

        Example:
            >>> processor = DonutProcessor(tokenizer=tokenizer)
            >>> inputs = processor.process_document("receipt.jpg", task_prompt="<s_cord-v2>")
            >>> # Use inputs with model for structured extraction
            >>> outputs = model.generate(**inputs)
        """
        # 1. Preprocess image
        pixel_values = self.process_image(image)

        # 2. Tokenize task prompt
        if self.tokenizer is not None:
            decoder_input_ids = self.tokenizer.encode(
                task_prompt,
                return_tensors="np",
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
            )
            decoder_input_ids = mx.array(decoder_input_ids)
        else:
            # Default BOS token if no tokenizer
            decoder_input_ids = mx.array([[0]])

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "task_prompt": task_prompt,
        }

    def extract_entities(
        self,
        generated_text: str,
        task_type: str = "cord-v2",
    ) -> dict:
        """
        Extract structured entities from Donut's generated text.

        Donut generates JSON-like output with special tokens marking entities.
        This function parses the generated text to extract structured data.

        Format Examples:
            - CORD-v2 (receipts):
              "<s_menu><s_nm>Burger</s_nm><s_price>$9.99</s_price></s_menu>"
            - DocVQA (Q&A):
              "<s_answer>The date is March 15, 2024</s_answer>"
            - RVLCDIP (classification):
              "<s_class>invoice</s_class>"

        Args:
            generated_text: Raw text from model generation
            task_type: Task type for parsing strategy
                - "cord-v2": Receipt parsing (menu items with names/prices)
                - "docvqa": Document question answering (extract answer)
                - "rvlcdip": Document classification (extract class)
                - "generic": Generic key-value extraction

        Returns:
            Dict with extracted entities. Structure depends on task_type:
                - cord-v2: {"menu": [{"name": str, "price": str}, ...]}
                - docvqa: {"answer": str}
                - rvlcdip: {"document_class": str}
                - generic: {key1: value1, key2: value2, ...}

        Example:
            >>> text = "<s_menu><s_nm>Coffee</s_nm><s_price>3.50</s_price></s_menu>"
            >>> entities = processor.extract_entities(text, "cord-v2")
            >>> print(entities)
            {'menu': [{'name': 'Coffee', 'price': '3.50'}]}
        """
        import re

        entities = {}

        if task_type == "cord-v2":
            # CORD (Consolidated Receipt Dataset) format
            # Extract menu items with names and prices
            menu_pattern = r'<s_menu>(.*?)</s_menu>'
            nm_pattern = r'<s_nm>(.*?)</s_nm>'
            price_pattern = r'<s_price>(.*?)</s_price>'

            menu_items = re.findall(menu_pattern, generated_text, re.DOTALL)
            entities['menu'] = []

            for item in menu_items:
                name_match = re.search(nm_pattern, item)
                price_match = re.search(price_pattern, item)

                if name_match and price_match:
                    entities['menu'].append({
                        'name': name_match.group(1).strip(),
                        'price': price_match.group(1).strip(),
                    })

            # Also extract other common receipt fields
            total_pattern = r'<s_total>(.*?)</s_total>'
            date_pattern = r'<s_date>(.*?)</s_date>'
            store_pattern = r'<s_store>(.*?)</s_store>'

            total_match = re.search(total_pattern, generated_text)
            if total_match:
                entities['total'] = total_match.group(1).strip()

            date_match = re.search(date_pattern, generated_text)
            if date_match:
                entities['date'] = date_match.group(1).strip()

            store_match = re.search(store_pattern, generated_text)
            if store_match:
                entities['store'] = store_match.group(1).strip()

        elif task_type == "docvqa":
            # Document VQA format - simple question-answer pairs
            answer_pattern = r'<s_answer>(.*?)</s_answer>'
            answer_match = re.search(answer_pattern, generated_text)

            if answer_match:
                entities['answer'] = answer_match.group(1).strip()
            else:
                # Fallback: if no special tokens, use entire text
                entities['answer'] = generated_text.strip()

        elif task_type == "rvlcdip":
            # Document classification format (RVL-CDIP dataset)
            # Classes: letter, form, email, handwritten, advertisement,
            #          scientific report, scientific publication, specification,
            #          file folder, news article, budget, invoice, presentation,
            #          questionnaire, resume, memo
            class_pattern = r'<s_class>(.*?)</s_class>'
            class_match = re.search(class_pattern, generated_text)

            if class_match:
                entities['document_class'] = class_match.group(1).strip()
            else:
                # Fallback: use entire text as class
                entities['document_class'] = generated_text.strip()

        else:
            # Generic key-value extraction
            # Matches patterns like: <s_key>value</s_key>
            kv_pattern = r'<s_([\w]+)>(.*?)</s_\1>'
            matches = re.findall(kv_pattern, generated_text)

            for key, value in matches:
                # Handle multiple values for same key
                if key in entities:
                    # Convert to list if not already
                    if not isinstance(entities[key], list):
                        entities[key] = [entities[key]]
                    entities[key].append(value.strip())
                else:
                    entities[key] = value.strip()

        return entities

    def __call__(self, text: str = None, image=None, **kwargs):
        """
        Process text and/or image.

        Args:
            text: Input text (optional)
            image: Input image (optional)
            **kwargs: Additional arguments

        Returns:
            Dictionary with processed inputs
        """
        outputs = {}

        if image is not None:
            outputs["pixel_values"] = self.process_image(image)

        if text is not None and self.tokenizer is not None:
            # Tokenize text
            tokens = self.tokenizer.encode(text, return_tensors="np")
            outputs["input_ids"] = mx.array(tokens)

        return outputs


def load_image(image_path: Union[str, Path, Image.Image]) -> Image.Image:
    """
    Load image from file or URL.

    Args:
        image_path: Path to image file, URL, or PIL Image object

    Returns:
        PIL Image

    Example:
        >>> image = load_image("document.jpg")
        >>> image = load_image("https://example.com/doc.jpg")
        >>> image = load_image(existing_image)  # Pass PIL Image directly
    """
    # If already a PIL Image, return as-is
    if isinstance(image_path, Image.Image):
        return image_path

    image_path = str(image_path)

    # Check if URL
    if image_path.startswith(("http://", "https://")):
        import requests
        from io import BytesIO

        response = requests.get(image_path)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        # Load from file and ensure it's fully loaded into memory
        with Image.open(image_path) as img:
            # Copy to ensure file handle is released
            image = img.copy()

    return image


def create_processor(image_size=(224, 224), tokenizer=None) -> DonutProcessor:
    """
    Create Donut processor with default settings.

    Args:
        image_size: Target image size
        tokenizer: Optional tokenizer

    Returns:
        DonutProcessor instance
    """
    return DonutProcessor(
        tokenizer=tokenizer,
        image_size=image_size,
        mean=(0.485, 0.456, 0.406),  # ImageNet mean
        std=(0.229, 0.224, 0.225),  # ImageNet std
    )
