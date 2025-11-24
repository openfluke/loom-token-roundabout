# loom-token-roundabout
loom-token-roundabout is a playful demo that trains and samples a tiny language model on a looping “roundabout” of forward → backward → accumulate → optimize steps. It builds a fresh BPE tokenizer from your text, runs a compact grid network with gradient accumulation and AdamW updates, and lets you spin out as many tokens as you like.

# Loom Smart-Batch Optimizer Demo

This demo is a self-contained example that trains a tiny language model with a smart-batching loop, gradient accumulation, a BPE tokenizer built on the fly, and the new pluggable optimizer framework (AdamW). It mirrors the earlier `s_b_s_d_v2` example but routes weight updates through `nn.NewAdamWOptimizer`, keeping embedding updates in a lightweight AdamW helper for the table.

## Repo name suggestion
- `loom-smart-batch-demo`
- `loom-step-optimizer-demo`
- `loom-bpe-smartbatch`

## What it does
- Builds a custom BPE tokenizer directly from the provided corpus (defaults to the `corpus/` text files or an Alice fallback).
- Slides a 32-token context window over the data, embeds tokens into 64-dim vectors, and feeds them into a small grid network: compress → scatter (guide + worker) → layer norm → mixer → vocab head.
- Accumulates gradients for `BatchSize` steps before applying them, using the framework AdamW for the network weights and a simple AdamW helper for embeddings.
- Periodically logs loss and generates sample text so you can watch training progress.

## Files
- `s_b_s_d_v2_optimizer.go` — main demo: data prep, tokenizer training, model build, training loop, and text generation.
- `README_smart_batch_optimizer.md` — this guide.

## How it works (high level)
1) **Tokenization**: Learns 500 BPE merges from the corpus, replaces spaces with `Ġ`, and builds `vocab.json` in memory (no files written).  
2) **Model**: A 1×5 grid with a compressor dense layer, a parallel scatter (guide dense + worker SwiGLU), layer norm, mixer dense, and vocab head.  
3) **Training**:  
   - Context window of 32 tokens; embedding dim 64.  
   - Gradients accumulate for `BatchSize=32` steps, then scaled and applied.  
   - Network updates use `nn.NewAdamWOptimizer`; embeddings use a lightweight AdamW helper.  
4) **Generation**: Temperature sampling (0.7) from the head logits to produce text continuations.

## Running it
```bash
cd examples/step_example
go run s_b_s_d_v2_optimizer.go
```

Options to tweak (edit the constants near the top of the file):
- `ContextSize`, `EmbeddingDim`, `BatchSize`, `LearningRate`, `WeightDecay`
- `compressDim`, `guideDim`, `workerDim`, `scatterOutput` inside the JSON config
- Number of BPE merges in `learnBPEAndGetConfig`

## How this differs from a traditional GPT-2
- Circular stepping vs. straight shot: this demo runs in a visible loop—forward, backward, stash gradients, step the optimizer, repeat—so you watch the cycle turn. GPT-2’s training is more of a single long conveyor belt from input to update.
- Swap-friendly controls: you pick the knobs (framework AdamW for the network, a tiny helper for embeddings) instead of a locked-in training routine.
- Fresh ingredients every run: the tokenizer is baked on the spot from whatever text you drop in, while GPT-2 ships with a fixed, pre-baked tokenizer.
- Lightweight rig: a compact grid that exists mainly to showcase the stepping loop, not a full Transformer stack.

## Expected output
You’ll see periodic logs like:
```
[Step 5000] Avg Loss: 6.24 | Pred: '_' Exp: 'down'
Gen: "Alicewaslybabb ..."
```
and a final generation after training.

## Notes and roadmap ideas
- Swap in different corpora by dropping `.txt` files into `examples/step_example/corpus/`.
- Experiment with schedulers (cosine/warmup) by wiring them through `nn` if you add a small config block.
- For larger vocab or context, watch memory: the embedding table grows with vocab size.
- Consider exporting checkpoints or tokenizer state if you extend this into a standalone repo.
