# MusicTransformer 🎵

**Status: Work in Progress** 🚧

This project is currently under development. We're building a music transformer model that can process and generate MIDI music data using advanced attention mechanisms.

## What We're Working On

- **MIDI Tokenization**: Converting MIDI files to token sequences and back
- **Music Generation**: Training transformer models on musical data with relative positional encoding
- **Audio Processing**: Handling different musical instruments and compositions

## Current Features

- ✅ MIDI to token conversion with 1ms time resolution
- ✅ Token to MIDI reconstruction
- ✅ Support for multiple instruments
- ✅ Velocity and timing preservation (128 velocity bins, no quantization loss)
- ✅ **Transformer model architecture with relative attention**
- ✅ **Relative positional encoding for musical sequences**
- ✅ **Multi-head attention with proper masking**

## Project Structure

```
MusicTransformer/
├── tokenizer.py          # MIDI ↔ Token conversion
├── relative_attention.py # Relative self-attention implementation
├── model.py             # Full transformer model
├── test_model.py        # Model testing script
├── example (1).mid      # Example MIDI file
└── README.md           # This file
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install torch pretty_midi numpy
   ```

2. **Test the tokenizer:**
   ```bash
   python tokenizer.py
   ```

3. **Test the model:**
   ```bash
   python test_model.py
   ```

## Model Architecture

- **Transformer with Relative Attention**: Uses relative positional encoding instead of absolute positions
- **Multi-head Attention**: 8 attention heads for capturing different musical patterns
- **Vocabulary Size**: 1384 tokens (notes, velocities, time shifts)
- **Time Resolution**: 1ms precision for accurate timing
- **Sequence Length**: Configurable (currently tested with 128 tokens)

## Tokenization Details

- **Note Events**: 128 pitches (C0 to G9)
- **Velocity**: 128 bins (1-127, preserving original MIDI velocity)
- **Time Shifts**: 1000 possible shifts (1ms to 1000ms resolution)
- **Total Vocabulary**: 1384 tokens

## TODO

- [ ] Add training pipeline
- [ ] Create evaluation metrics for music quality
- [ ] Implement music generation sampling strategies
- [ ] Add support for different musical styles
- [ ] Model checkpointing and saving
- [ ] Data preprocessing for large MIDI datasets
- [ ] Implement conditioning for controlled generation
- [ ] Add visualization tools for attention patterns

## Contributors

- **Rohith** 
- **Ananthannn** 

## Contributing

This project is actively being developed. The core architecture is now functional! Feel free to explore the code and contribute ideas for training and evaluation.

---

*More documentation and training examples coming soon as we continue development...*
