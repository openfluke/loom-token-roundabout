package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"

	nn "github.com/openfluke/loom/nn"
	tokenizer "github.com/openfluke/loom/tokenizer"
)

// --- Configuration ---
const (
	ContextSize  = 48
	EmbeddingDim = 16
	BatchSize    = 8
	LearningRate = 0.002
	WeightDecay  = 0.01
	ClipNorm     = 1.0
	WarmupSteps  = 1000
)

// --- Embedding Optimizer (lightweight AdamW for tables only) ---
type EmbeddingAdamW struct {
	beta1, beta2, epsilon, weightDecay float32
	m, v                               map[int][]float32
	step                               int
}

func NewEmbeddingAdamW(beta1, beta2, epsilon, weightDecay float32) *EmbeddingAdamW {
	return &EmbeddingAdamW{
		beta1:       beta1,
		beta2:       beta2,
		epsilon:     epsilon,
		weightDecay: weightDecay,
		m:           make(map[int][]float32),
		v:           make(map[int][]float32),
	}
}

func (opt *EmbeddingAdamW) Step(lr float32, emb *EmbeddingTable, batchSize int) {
	opt.step++
	scale := float32(1.0) / float32(batchSize)

	biasCorrection1 := 1.0 - float32(math.Pow(float64(opt.beta1), float64(opt.step)))
	biasCorrection2 := 1.0 - float32(math.Pow(float64(opt.beta2), float64(opt.step)))

	for row := range emb.Weights {
		if len(emb.Weights[row]) == 0 {
			continue
		}
		if opt.m[row] == nil {
			opt.m[row] = make([]float32, len(emb.Weights[row]))
			opt.v[row] = make([]float32, len(emb.Weights[row]))
		}

		for j := range emb.Weights[row] {
			g := emb.Grads[row][j] * scale

			opt.m[row][j] = opt.beta1*opt.m[row][j] + (1-opt.beta1)*g
			opt.v[row][j] = opt.beta2*opt.v[row][j] + (1-opt.beta2)*g*g

			mHat := opt.m[row][j] / biasCorrection1
			vHat := opt.v[row][j] / biasCorrection2

			emb.Weights[row][j] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+opt.epsilon) + opt.weightDecay*emb.Weights[row][j])
			emb.Grads[row][j] = 0
		}
	}
}

// --- Embedding Table (with accumulation) ---
type EmbeddingTable struct {
	Weights [][]float32
	Grads   [][]float32
	Dim     int
}

func NewEmbeddingTable(vocabSize, dim int) *EmbeddingTable {
	weights := make([][]float32, vocabSize)
	grads := make([][]float32, vocabSize)
	scale := float32(math.Sqrt(2.0 / float64(dim)))

	for i := range weights {
		weights[i] = make([]float32, dim)
		grads[i] = make([]float32, dim)
		for j := range weights[i] {
			weights[i][j] = (rand.Float32()*2 - 1) * scale
		}
	}
	return &EmbeddingTable{Weights: weights, Grads: grads, Dim: dim}
}

func (e *EmbeddingTable) Forward(window []int) []float32 {
	out := make([]float32, len(window)*e.Dim)
	for i, id := range window {
		if id < len(e.Weights) {
			copy(out[i*e.Dim:], e.Weights[id])
		}
	}
	return out
}

func (e *EmbeddingTable) Accumulate(window []int, gradInput []float32) {
	for i, id := range window {
		if id < len(e.Grads) {
			start := i * e.Dim
			end := start + e.Dim
			chunk := gradInput[start:end]
			for j := 0; j < e.Dim; j++ {
				e.Grads[id][j] += chunk[j]
			}
		}
	}
}

// --- Byte-level BPE config (no training) ---
func buildByteLevelBPEConfig() string {
	vocab := make(map[string]int)
	idx := 0
	vocab["<UNK>"] = idx
	idx++
	vocab["<PAD>"] = idx
	idx++

	for b := 0; b < 256; b++ {
		token := fmt.Sprintf("<0x%02X>", b)
		vocab[token] = idx
		idx++
	}

	type TokenizerModel struct {
		Type         string         `json:"type"`
		Vocab        map[string]int `json:"vocab"`
		Merges       []string       `json:"merges"`
		ByteFallback bool           `json:"byte_fallback"`
	}
	config := struct {
		Model       TokenizerModel `json:"model"`
		AddedTokens []struct{}     `json:"added_tokens"`
	}{
		Model: TokenizerModel{
			Type:         "BPE",
			Vocab:        vocab,
			Merges:       []string{},
			ByteFallback: true,
		},
		AddedTokens: []struct{}{},
	}

	b, _ := json.Marshal(config)
	return string(b)
}

// --- Generation ---
func generate(net *nn.Network, embs *EmbeddingTable, tok *tokenizer.Tokenizer, seed string, maxTokens int) string {
	window := make([]int, ContextSize)
	for i := range window {
		window[i] = 1 // PAD
	}

	seedIDs := tok.Encode(seed, false)
	for _, id := range seedIDs {
		copy(window, window[1:])
		window[ContextSize-1] = int(id)
	}

	state := net.InitStepState(ContextSize * EmbeddingDim)

	freq := make(map[int]int)
	var allIDs []uint32
	allIDs = append(allIDs, seedIDs...)

	for i := 0; i < maxTokens; i++ {
		inputVec := embs.Forward(window)
		state.SetInput(inputVec)
		net.StepForward(state)
		out := state.GetOutput()

		maxVal := out[0]
		for _, v := range out {
			if v > maxVal {
				maxVal = v
			}
		}
		exps := make([]float32, len(out))
		temperature := 0.9
		repetitionPenalty := 1.05
		for j, v := range out {
			if j == 0 || j == 1 { // <UNK>/<PAD>
				exps[j] = 0
				continue
			}
			score := math.Exp(float64((v - maxVal) / float32(temperature)))
			if c := freq[j]; c > 0 {
				score /= math.Pow(repetitionPenalty, float64(c))
			}
			exps[j] = float32(score)
		}

		type candidate struct {
			idx int
			p   float32
		}
		cands := make([]candidate, 0, len(exps))
		for j, p := range exps {
			if p > 0 {
				cands = append(cands, candidate{idx: j, p: p})
			}
		}
		if len(cands) == 0 {
			break
		}
		topK := 30
		if len(cands) < topK {
			topK = len(cands)
		}
		sort.Slice(cands, func(a, b int) bool { return cands[a].p > cands[b].p })
		cands = cands[:topK]

		sumTop := float32(0)
		for _, c := range cands {
			sumTop += c.p
		}

		best := cands[0].idx
		if sumTop > 0 {
			r := rand.Float32() * sumTop
			cum := float32(0)
			for _, c := range cands {
				cum += c.p
				if cum >= r {
					best = c.idx
					break
				}
			}
		}

		freq[best]++
		allIDs = append(allIDs, uint32(best))

		copy(window, window[1:])
		window[ContextSize-1] = best
	}

	decoded := tok.Decode(allIDs, true)
	return decoded
}

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("=== LOOM: LSTM Demo (BPE tokenizer) ===")

	// 1. Load Corpus
	corpusDir := "./corpus"
	var sb strings.Builder
	files, _ := os.ReadDir(corpusDir)
	if len(files) > 0 {
		for _, f := range files {
			if filepath.Ext(f.Name()) == ".txt" {
				b, _ := os.ReadFile(filepath.Join(corpusDir, f.Name()))
				sb.Write(b)
				sb.WriteString(" ")
			}
		}
	}
	textData := sb.String()
	if len(textData) == 0 {
		fmt.Println("Using fallback text.")
		base := "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do."
		for i := 0; i < 200; i++ {
			textData += " " + base
		}
	}

	// 2. Byte-level BPE tokenizer (no training)
	jsonConfig := buildByteLevelBPEConfig()
	tok, err := tokenizer.LoadFromBytes([]byte(jsonConfig))
	if err != nil {
		log.Fatalf("failed to load tokenizer: %v", err)
	}
	vocabSize := len(tok.Vocab)
	fmt.Printf("Vocab Size: %d\n", vocabSize)

	// 3. Encode
	rawIDs := tok.Encode(strings.ReplaceAll(textData, " ", "Ġ"), false)
	dataIDs := make([]int, len(rawIDs))
	for i, v := range rawIDs {
		dataIDs[i] = int(v)
	}
	fmt.Printf("Total Tokens: %d\n", len(dataIDs))

	// 4. Network Construction
	flatInputSize := ContextSize * EmbeddingDim
	embTable := NewEmbeddingTable(vocabSize, EmbeddingDim)

	compressDim := 64
	guideDim := 4
	workerDim := 64
	memoryDim := 32
	feedbackDim := 8
	scatterOutput := guideDim + workerDim + memoryDim + feedbackDim
	mixerHidden := 64

	networkJSON := fmt.Sprintf(`{
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 5, 
        "layers_per_cell": 1,
        "layers": [
            { 
                "type": "dense", 
                "id": "compressor",
                "input_height": %d, 
                "output_height": %d, 
                "activation": "tanh" 
            },
            {
                "type": "parallel",
                "id": "scatter_core",
                "combine_mode": "grid_scatter",
                "grid_output_rows": 1,
                "grid_output_cols": 4,
                "grid_output_layers": 1,
                "grid_positions": [
                    {"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
                    {"branch_index": 1, "target_row": 0, "target_col": 1, "target_layer": 0},
                    {"branch_index": 2, "target_row": 0, "target_col": 2, "target_layer": 0},
                    {"branch_index": 3, "target_row": 0, "target_col": 3, "target_layer": 0}
                ],
                "branches": [
                    { 
                        "type": "dense", 
                        "id": "guide",
                        "input_height": %d, 
                        "output_height": %d, 
                        "activation": "linear" 
                    },
                    { 
                        "type": "lstm", 
                        "id": "worker",
                        "input_size": %d, 
                        "hidden_size": %d,
                        "seq_length": 1
                    },
                    {
                        "type": "dense",
                        "id": "memory_refiner",
                        "input_height": %d, 
                        "output_height": %d, 
                        "activation": "tanh" 
                    },
                    {
                        "type": "dense", 
                        "id": "feedback_compressor",
                        "input_height": %d, 
                        "output_height": %d, 
                        "activation": "linear" 
                    }
                ]
            },
            {
                "type": "layer_norm",
                "id": "norm",
                "norm_size": %d,
                "epsilon": 1e-5
            },
            { 
                "type": "parallel", 
                "id": "mixer_block",
                "combine_mode": "add",
                "branches": [
                    {
                        "type": "lstm", 
                        "id": "mixer",
                        "input_size": %d, 
                        "hidden_size": %d,
                        "seq_length": 1
                    },
                    {
                        "type": "dense", 
                        "id": "mixer_skip",
                        "input_height": %d, 
                        "output_height": %d, 
                        "activation": "linear" 
                    }
                ]
            },
            { 
                "type": "dense", 
                "id": "head",
                "input_height": %d, 
                "output_height": %d, 
                "activation": "linear" 
            }
        ]
	}`,
		flatInputSize, compressDim,
		compressDim, guideDim,
		compressDim, workerDim,
		compressDim, memoryDim,
		compressDim, feedbackDim,
		scatterOutput,
		scatterOutput, mixerHidden,
		scatterOutput, mixerHidden,
		mixerHidden, vocabSize)

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatal(err)
	}
	net.InitializeWeights()
	net.SetOptimizer(nn.NewAdamWOptimizer(0.9, 0.999, 1e-8, WeightDecay))
	state := net.InitStepState(flatInputSize)

	// Gradient accumulators
	accKernels := make([][]float32, len(net.Layers))
	accBiases := make([][]float32, len(net.Layers))
	for i, layer := range net.Layers {
		if len(layer.Kernel) > 0 {
			accKernels[i] = make([]float32, len(layer.Kernel))
		}
		if len(layer.Bias) > 0 {
			accBiases[i] = make([]float32, len(layer.Bias))
		}
	}

	embOptimizer := NewEmbeddingAdamW(0.9, 0.999, 1e-8, WeightDecay)

	// 5. Training Loop
	steps := 150000
	ptr := 0
	batchCounter := 0
	epochLoss := 0.0

	window := make([]int, ContextSize)
	for i := range window {
		window[i] = 1 // PAD
	}

	fmt.Println("Training with AdamW + Batch Accumulation (BPE tokens)...")
	start := time.Now()

	for i := 0; i < steps; i++ {
		if ptr+1 >= len(dataIDs) {
			ptr = 0
			for k := range window {
				window[k] = 1
			}
		}

		target := dataIDs[ptr]

		// Forward pass
		inputVec := embTable.Forward(window)
		state.SetInput(inputVec)
		net.StepForward(state)
		out := state.GetOutput()

		// Loss + grad (cross-entropy)
		grad := make([]float32, vocabSize)
		maxVal := out[0]
		for _, v := range out {
			if v > maxVal {
				maxVal = v
			}
		}
		sumExp := float32(0.0)
		exps := make([]float32, vocabSize)
		for j, v := range out {
			exps[j] = float32(math.Exp(float64(v - maxVal)))
			sumExp += exps[j]
		}

		stepLoss := float32(0.0)
		for j := range out {
			prob := exps[j] / sumExp
			t := float32(0.0)
			if j == target {
				t = 1.0
			}
			grad[j] = prob - t
			if t > 0.5 {
				stepLoss -= float32(math.Log(float64(prob + 1e-9)))
			}
		}
		epochLoss += float64(stepLoss)

		// Backward
		gradInput, _ := net.StepBackward(state, grad)

		// Accumulate
		embTable.Accumulate(window, gradInput)

		currentKernelGrads := net.KernelGradients()
		currentBiasGrads := net.BiasGradients()

		for lIdx := range net.Layers {
			if len(accKernels[lIdx]) > 0 && len(currentKernelGrads[lIdx]) > 0 {
				for k := range accKernels[lIdx] {
					accKernels[lIdx][k] += currentKernelGrads[lIdx][k]
				}
			}
			if len(accBiases[lIdx]) > 0 && len(currentBiasGrads[lIdx]) > 0 {
				for k := range accBiases[lIdx] {
					accBiases[lIdx][k] += currentBiasGrads[lIdx][k]
				}
			}
		}

		// Slide window
		copy(window, window[1:])
		window[ContextSize-1] = target
		ptr++
		batchCounter++

		// Optimizer step
		if batchCounter >= BatchSize {
			lr := lrSchedule(i, steps, LearningRate)

			clipNetworkAccums(accKernels, accBiases, ClipNorm)
			clipEmbeddingGrads(embTable, ClipNorm)

			scale := float32(1.0) / float32(BatchSize)
			netKernelGrads := net.KernelGradients()
			netBiasGrads := net.BiasGradients()

			for lIdx := range net.Layers {
				if len(accKernels[lIdx]) > 0 && len(netKernelGrads[lIdx]) > 0 {
					for k := range netKernelGrads[lIdx] {
						netKernelGrads[lIdx][k] = accKernels[lIdx][k] * scale
						accKernels[lIdx][k] = 0
					}
				}
				if len(accBiases[lIdx]) > 0 && len(netBiasGrads[lIdx]) > 0 {
					for k := range netBiasGrads[lIdx] {
						netBiasGrads[lIdx][k] = accBiases[lIdx][k] * scale
						accBiases[lIdx][k] = 0
					}
				}
			}

			embOptimizer.Step(lr, embTable, BatchSize)
			net.ApplyGradients(lr)

			batchCounter = 0
		}

		if i > 0 && i%5000 == 0 {
			avgLoss := epochLoss / 5000.0
			epochLoss = 0.0

			fmt.Println("\n------------------------------------------------------------")
			fmt.Printf(" [Step %d] Avg Loss: %.4f\n", i, avgLoss)
			fmt.Printf(" Gen: \"%s...\"\n", generate(net, embTable, tok, " Alice", 30))
			fmt.Println("------------------------------------------------------------")
		}
	}

	fmt.Printf("Training Complete in %v\n", time.Since(start))
	fmt.Println("\n=== FINAL GENERATION ===")
	fmt.Println(generate(net, embTable, tok, " Alice", 50))
}

// --- Helpers ---
func lrSchedule(step, totalSteps int, baseLR float32) float32 {
	if totalSteps <= WarmupSteps {
		return baseLR
	}
	if step < WarmupSteps {
		return baseLR * float32(step+1) / float32(WarmupSteps)
	}
	progress := float64(step-WarmupSteps) / float64(totalSteps-WarmupSteps)
	return baseLR * 0.5 * float32(1.0+math.Cos(math.Pi*progress))
}

func clipNetworkAccums(accKernels, accBiases [][]float32, maxNorm float32) {
	var sum float64
	for _, g := range accKernels {
		for _, v := range g {
			sum += float64(v * v)
		}
	}
	for _, g := range accBiases {
		for _, v := range g {
			sum += float64(v * v)
		}
	}
	norm := math.Sqrt(sum)
	if norm == 0 || norm <= float64(maxNorm) {
		return
	}
	scale := maxNorm / float32(norm)
	for _, g := range accKernels {
		for i := range g {
			g[i] *= scale
		}
	}
	for _, g := range accBiases {
		for i := range g {
			g[i] *= scale
		}
	}
}

func clipEmbeddingGrads(emb *EmbeddingTable, maxNorm float32) {
	var sum float64
	for i := range emb.Grads {
		for _, v := range emb.Grads[i] {
			sum += float64(v * v)
		}
	}
	norm := math.Sqrt(sum)
	if norm == 0 || norm <= float64(maxNorm) {
		return
	}
	scale := maxNorm / float32(norm)
	for i := range emb.Grads {
		for j := range emb.Grads[i] {
			emb.Grads[i][j] *= scale
		}
	}
}

func tokenLooksLikeWord(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}
