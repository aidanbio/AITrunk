"""
This modules is used for extracting text encodings using pretrained CLIP model that trained on image-text pairs.
The model card can be found https://huggingface.co/openai/clip-vit-large-patch32. The CLIP paper can be found at https://arxiv.org/abs/2103.00020
"""
import unittest

from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        """
        :param model_id: is the model id
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        # Load the tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        # Load the CLIP transformer
        self.transformer = CLIPTextModel.from_pretrained(model_id).eval()

    @property
    def hidden_size(self):
        return self.transformer.config.hidden_size

    @property
    def max_length(self):
        return self.transformer.config.max_position_embeddings

    def empty_encoded(self, batch_size):
        return self.forward([''] * batch_size)

    def forward(self, prompts=None):
        """
        :param prompts: are the list of prompts to be encoded
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.transformer.device)
        return self.transformer(input_ids=tokens).last_hidden_state

class CLIPTextEncoderTest(unittest.TestCase):
    def setUp(self):
        self.text_encoder = CLIPTextEncoder()

    def test_forward(self):
        prompts = ["a photo of a cat", "a photo of a dog"]
        encoded = self.text_encoder(prompts)
        self.assertEqual(encoded.shape, (2, self.text_encoder.max_length, self.text_encoder.hidden_size))