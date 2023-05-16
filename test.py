import unittest
import torch

from base.bert_base import BertEmbedding
from base.transformer_base import (
    MhaBlock,
    PositionWiseFFN,
    EncoderBlock,
    Encoder,
    Decoder,
)


class TestModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_len = 128
        self.d_model = 768
        self.n_heads = 12
        self.n_tokens = 30522  # Vocabulary size for BERT Base
        self.d_ff = 3072  # Feed-forward hidden size for BERT Base
        self.n_layers = 12  # Number of layers in BERT Base

    def test_BertEmbedding(self):
        x = torch.randint(self.n_tokens, (self.batch_size, self.seq_len))
        model = BertEmbedding(self.n_tokens, self.d_model)
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_MhaBlock(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        model = MhaBlock(self.d_model, self.n_heads)
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_PositionWiseFFN(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        model = PositionWiseFFN(self.d_model, self.d_ff)
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_EncoderBlock(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        model = EncoderBlock(self.d_model, self.n_heads)
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_Encoder(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        model = Encoder(self.d_model, self.n_layers, self.n_heads)
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))


class TestMasking(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.n_heads = 4
        self.seq_len = 10
        self.batch_size = 1

    def test_MhaBlock_masking(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        model = MhaBlock(self.d_model, self.n_heads)

        # Set a mask that only allows each position to attend to previous positions
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))

        out1 = model(x, mask=mask)

        # Now modify the input at the last position
        x_prime = x.clone()
        x_prime[0, -1, :] += 1.0

        out2 = model(x_prime, mask=mask)

        # Check that the output at all positions except the last is unchanged
        self.assertTrue(torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-7))


class TestEncoderDecoder(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.n_heads = 4
        self.seq_len = 10
        self.batch_size = 1
        self.n_layers = 2

    def test_encoder_decoder_attention(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        encoder = Encoder(self.d_model, self.n_layers, self.n_heads)
        decoder = Decoder(self.d_model, self.n_layers, self.n_heads)

        # Create a 2D mask for the decoder
        mask = torch.ones(self.batch_size, self.seq_len)

        # Run the input through the encoder and decoder
        encoder_out = encoder(x)
        out1 = decoder(encoder_out, mask)

        # Now modify the input and run it through the encoder and decoder again
        x_prime = x.clone()
        x_prime[0, 0, :] += 1.0  # Modify the first position
        encoder_out_prime = encoder(x_prime)
        out2 = decoder(encoder_out_prime, mask)

        # Check that the output has changed
        self.assertFalse(torch.allclose(out1, out2, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
