"""
Protein sequence encoder using pre-trained pLMs
"""
import unittest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

from tcrdiffusion.bioseq import UNKNOWN, rand_aaseqs


class ProteinSeqEncoder(nn.Module):
    def __init__(self, plm_id_or_path='facebook/esm2_t33_650M_UR50D'):
        super().__init__()
        self.plm = AutoModelForMaskedLM.from_pretrained(plm_id_or_path, output_hidden_states=True).eval()
        # self.plm = AutoModelForMaskedLM.from_pretrained(plm_id_or_path, output_hidden_states=True, output_attentions=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(plm_id_or_path)

    @property
    def hidden_size(self):
        return self.plm.config.hidden_size

    def unknown_to_mask(self, seqs):
        new_seqs = []
        for seq in seqs:
            for unk in UNKNOWN:
                seq = seq.replace(unk, self.tokenizer.mask_token)
            new_seqs.append(seq)
        return new_seqs

    @torch.no_grad()
    def encode(self, seqs, max_length=15):
        """
        :param seqs: are the list of sequences to be encoded
        :param max_length: is the max length of the tokenized sequences
        """
        seqs = self.unknown_to_mask(seqs)
        encoded = self.tokenizer(seqs,
                                 padding="max_length",
                                 truncation=False,
                                 max_length=max_length,
                                 return_overflowing_tokens=False,
                                 return_tensors="pt")
        outputs = self.plm(**encoded)
        return outputs.hidden_states[-1]

    @torch.no_grad()
    def decode(self, hidden_states):
        """
        :param hidden_states: are the hidden states, shape: (batch_size, max_length, hidden_size)
        :return:
        """
        logits = self.plm.lm_head(hidden_states)
        token_ids = torch.argmax(logits, dim=-1)
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def forward(self, seqs, max_length=15):
        return self.encode(seqs, max_length=max_length)


class ProteinSeqEncoderTest(unittest.TestCase):
    def setUp(self):
        self.seq_encoder = ProteinSeqEncoder()

    def test_unknown_to_mask(self):
        seqs = ['CATSRERAGGGTDTQYF', 'ATSRER*GGGTDTQYF', 'TSRERAG-*GTDTQYF', 'SRERAGGGTDTQYF', 'RERAGG*TDTQYF']
        new_seqs = self.seq_encoder.unknown_to_mask(seqs)
        mask_token = self.seq_encoder.tokenizer.mask_token
        self.assertListEqual(seqs, ['CATSRERAGGGTDTQYF', 'ATSRER*GGGTDTQYF', 'TSRERAG-*GTDTQYF', 'SRERAGGGTDTQYF', 'RERAGG*TDTQYF'])
        self.assertEqual(new_seqs, ['CATSRERAGGGTDTQYF', f'ATSRER{mask_token}GGGTDTQYF', f'TSRERAG{mask_token}{mask_token}GTDTQYF', 'SRERAGGGTDTQYF', f'RERAGG{mask_token}TDTQYF'])

    def test_encode(self):
        epitopes = ['MIELSLIDFY', 'IELSLIDFYL', 'ELSLIDFYLL', 'LSLIDFYLLN', 'SLIDFYLLNL']

        max_length = 15
        encoded = self.seq_encoder(epitopes, max_length=max_length)
        self.assertTrue(isinstance(encoded, torch.Tensor))
        self.assertEqual(encoded.shape, (len(epitopes), max_length, self.seq_encoder.hidden_size))

        max_length = 25
        cdr3bs = ['CATSRERAGGGTDTQYF', 'ATSRER*GGGTDTQYF', 'TSRERAG-*GTDTQYF', 'SRERAGGGTDTQYF', 'RERAGG*TDTQYF']
        encoded = self.seq_encoder(cdr3bs, max_length=max_length)
        self.assertTrue(isinstance(encoded, torch.Tensor))
        self.assertEqual(encoded.shape, (len(cdr3bs), max_length, self.seq_encoder.hidden_size))

    def test_decode(self):
        # epitopes = ['MIELSLIDFY', 'IELSLIDFYL', 'ELSLIDFYLL', 'LSLIDFYLLN', 'SLIDFYLLNL']
        epitopes = rand_aaseqs(10, 9)
        max_length = 15
        encoded = self.seq_encoder(epitopes, max_length=max_length)
        self.assertTrue(isinstance(encoded, torch.Tensor))
        self.assertEqual(encoded.shape, (len(epitopes), max_length, self.seq_encoder.hidden_size))

        decoded = self.seq_encoder.decode(encoded)
        self.assertEqual(decoded, epitopes)

        max_length = 25
        cdr3bs = ['CATSRERAGGGTDTQYF', 'ATSRER*GGGTDTQYF', 'TSRERAG-*GTDTQYF', 'SRERAGGGTDTQYF', 'RERAGG*TDTQYF']
        encoded = self.seq_encoder(cdr3bs, max_length=max_length)
        self.assertTrue(isinstance(encoded, torch.Tensor))
        self.assertEqual(encoded.shape, (len(cdr3bs), max_length, self.seq_encoder.hidden_size))

if __name__ == '__main__':
    unittest.main()
