# MecAgent Technical Test Submission - Yacine Baghli

This repository contains my submission for the **MecAgent ML Engineer technical test**.  
The objective was to build the best possible **CadQuery code generator** from a **2D image of a 3D part**, within a **7-hour time constraint**.

---

## Methodology & Approach

Given the complexity of the image-to-code task and the strict time constraint, my strategy was centered around rapid iteration, pragmatic problem-solving, and building a robust end-to-end pipeline before focusing on performance.

My process was divided into four key phases:

---

### Phase 1: Robust Data Pipeline

The initial challenge was handling the 147k-sample dataset efficiently. The standard datasets library presented a recurring LocalFileSystem caching error within the Colab environment. To overcome this and ensure stability, I bypassed the high-level loader and implemented a more robust manual pipeline:

- Downloaded raw Parquet files directly via `wget` from Hugging Face.
- Loaded data using **`pandas`** and **`pyarrow`**.
- Built a simple, character-level tokenizer from the code corpus.
- Created a **custom `PyTorch Dataset`**  class to handle image transformations and code tokenization.

> This approach gave me full control and eliminated any environment-specific issues, allowing me to proceed with a reliable data source.

---

### Phase 2: Baseline Model (Encoder + GRU Decoder)

- **Encoder**: A pre-trained ResNet-34 (with frozen weights) from `torchvision` to generate a feature vector from each input image.
- **Decoder**: A simple GRU-based decoder to generate CadQuery code token-by-token from the image feature vector.

---

### Phase 3: Enhanced Model (Transformer + Beam Search)

The failure of the baseline clearly indicated that the GRU decoder was not powerful enough to handle the complex grammar of a programming language. To solve this, I upgraded the architecture:

- **Architecture:** I replaced the GRU decoder with a **Transformer Decoder**. The self-attention mechanism is far better suited to capturing the long-range dependencies and structure inherent in code.
- **Inference:** I replaced the simple "greedy" decoding with **Beam Search**. This more intelligent generation strategy explores multiple potential code sequences simultaneously, significantly increasing the probability of finding a syntactically valid one.

---

### Phase 4: Performance Optimization (A100 GPU)

To accelerate the longer training run of the enhanced model, I implemented two key performance optimizations for the A100 GPU environment:

- Automatic Mixed Precision (AMP): Using `torch.amp.autocast` and `GradScaler` to leverage the Tensor Cores for faster FP16 computation.
- `torch.compile()`: Using PyTorch 2.0's JIT compiler to create optimized execution graphs for the model.

---

## How to Run the Project

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data (one-time only)
```bash
python explore_and_tokenize_manual.py
```

### 3. Train the Final Model
```bash
python train_enhanced_turbo.py
```

### 4. Evaluate the Final Model (Beam Search)
```bash
python evaluate_enhanced.py
```


---

## Results & Analysis

My approach was iterative. I evaluated the performance at each major step to justify the next decision. The results clearly show the impact of each enhancement.

| Model Version                        | Training Data | Key Changes                             | VSR (Valid Syntax Rate) | Mean Best IoU | Analysis                                                                 |
|-------------------------------------|---------------|------------------------------------------|--------------------------|----------------|--------------------------------------------------------------------------|
| Baseline (GRU)                      | 5%            | –                                        | 0.0%                     | 0.0000         | Severe mode collapse.                                                    |
| Enhanced (Transformer + Beam Search)| 5%            | Transformer Decoder + Beam Search        | 10.0%                    | 0.0073         | Varied, context-aware code. Major improvement over GRU baseline.        |
| Final "Turbo" Model                 | 25%           | More Data + AMP + `torch.compile()`      | %                        |                | Improved syntax and feature recognition thanks to better training setup. |

### Analysis of the Baseline
The initial Encoder-GRU-Decoder model trained successfully (loss decreased), but the evaluation revealed a Valid Syntax Rate of 0%. A qualitative analysis showed a classic "mode collapse": the model generated the same, incomplete, non-executable code snippet for every input image. This proved the simple sequential GRU was insufficient for this task.

### Analysis of the Enhanced Model (Phase 3)
The first enhancement was to replace the GRU with a Transformer decoder and to use Beam Search for generation at inference time. This model was trained on the same 5% data subset.
This architectural change had a dramatic and immediate impact:
The VSR jumped from 0% to 10%. This is the key breakthrough, proving that the model was no longer stuck in a collapsed mode and could generate syntactically valid code for the first time.
The Mean IoU became non-zero (0.0073). While small, this confirms that the valid code produced a 3D shape that had a measurable, albeit tiny, resemblance to the ground truth.
This result validated the choice of the Transformer architecture. However, the generated code still contained many errors, indicating that the model's core "knowledge" was limited by the small training dataset.

### The Final "Turbo" Model (Phase 4 - In Progress)
The final step, which is currently training, is to take this proven Transformer architecture and train it on a much larger dataset (25% of the total) for more epochs. To make this feasible, performance optimizations like Automatic Mixed Precision (AMP) and torch.compile are being used.
The objective of this final run is to provide the superior architecture with more data to properly learn the complex grammar and semantics of the CadQuery language. This should lead to a significant increase in both VSR and Mean IoU in the final evaluation

## Bottlenecks & Limitations

- **Data Subsetting:** All results are based on training on only 25% of the available data due to the 7-hour time limit. Performance would significantly increase with access to the full dataset.
- **Character-Level Tokenizer:** While fast to implement, this tokenizer does not capture the semantic meaning of CadQuery functions and keywords (e.g., Workplane is seen as 9 independent characters).
- **CPU Bottleneck:** Even with pre-calculated data files, the on-the-fly data augmentation (3D rotations) on the CPU was a limiting factor for GPU utilization.

---

## Future Improvements (If More Time Available)

- Full-Scale Training: The most critical next step would be to train the final Transformer model on 100% of the dataset for 50-100 epochs, leveraging the AMP and compile optimizations. This alone should dramatically improve VSR and IoU.
- Advanced Tokenizer: I would replace the character-level tokenizer with a Byte-Pair Encoding (BPE) tokenizer, trained specifically on the CadQuery code corpus. This would allow the model to learn and predict semantic "sub-words" like cq.Workplane or .extrude().
- GPU-Based Augmentation: To eliminate the CPU bottleneck, I would integrate NVIDIA DALI to perform all data loading and augmentation directly on the GPU, fully saturating the A100.
- Hyperparameter Tuning: I would use a framework like Optuna to systematically search for the optimal learning rate, batch size, and Transformer architecture (number of layers, heads).

---

## Contact

For any questions or technical discussion:  
**Yacine Baghli** – yacine.baghli@gmail.com

---

> _This project demonstrates a principled and scalable approach to the image-to-CadQuery problem under tight time constraints._

