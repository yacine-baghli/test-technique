# MecAgent Technical Test Submission - Yacine Baghli

This repository contains my submission for the **MecAgent ML Engineer technical test**.  
The objective was to build the best possible **CadQuery code generator** from a **2D image of a 3D part**, within a **7-hour time constraint**.

## Key Achievements
- Successfully built and demonstrated a complete end-to-end Image-to-Code pipeline, from data processing to training, optimization, and evaluation.
- Systematically improved model performance, increasing the Valid Syntax Rate (VSR) from 0% to 70%.
- Diagnosed and solved multiple real-world engineering challenges, including data pipeline errors, model "mode collapse", and GPU Out-of-Memory errors.
- Implemented advanced techniques such as Transformer architecture, Beam Search decoding, and GPU optimizations (AMP, torch.compile).

---

## Methodology & Approach

Given the complexity of the image-to-code task and the strict time constraint, my strategy was centered around rapid iteration, pragmatic problem-solving, and building a robust end-to-end pipeline before focusing on performance.

My process was divided into four key phases:

---

## My Development Journey
Given the complexity of the task and the strict time limit, my strategy was centered around rapid, iterative development.

### Phase 1: Robust Data Pipeline
The initial challenge was the dataset's instability in the Colab environment. I engineered a robust solution by bypassing the high-level library, downloading the raw Parquet files manually with `wget`, and loading them with `pandas`. This provided a stable foundation for the project. A character-level tokenizer was then built from this data.

### Phase 2: The Baseline (Encoder-GRU Decoder)
I first established a baseline using a classic Encoder-Decoder architecture (frozen ResNet-34 + GRU Decoder). This allowed me to build and verify the full training and evaluation pipeline quickly. As anticipated, this simple model suffered from severe mode collapse, resulting in a VSR of 0%.

### Phase 3: The Enhanced Model (Transformer + Beam Search)
To solve the mode collapse, I upgraded the architecture by replacing the GRU with a Transformer Decoder. At inference time, I also implemented Beam Search decoding instead of a simple greedy search. This proved to be the critical breakthrough, raising the VSR from 0% to 10% on an initial training run.

### Phase 4: The Final "Turbo" Model
To maximize performance, the proven Transformer architecture was trained on a larger data subset (25%) for more epochs. To make this feasible, the training script was optimized for the A100 GPU using Automatic Mixed Precision (AMP) and `torch.compile()`. This produced the final model submitted for evaluation.

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
python evaluate_enhanced_turbo.py
```


---

## Results & Analysis

My approach was iterative. I evaluated the performance at each major step to justify the next decision. The results clearly show the impact of each enhancement.

| Model Version                        | Training Data | Key Changes                             | VSR (Valid Syntax Rate) | Mean Best IoU | Analysis                                                                 |
|-------------------------------------|---------------|------------------------------------------|--------------------------|----------------|--------------------------------------------------------------------------|
| Baseline (GRU)                      | 5%            | –                                        | 0.0%                     | 0.00%          | Severe mode collapse.                                                    |
| Enhanced (Transformer + Beam Search)| 5%            | Transformer Decoder + Beam Search        | 10.0%                    | 0.73%          | Varied, context-aware code. Major improvement over GRU baseline.        |
| Final "Turbo" Model                 | 25%           | More Data + AMP + `torch.compile()`      | 70.0%                    | 19.93%         | Improved syntax and feature recognition thanks to better training setup. |


The final model demonstrates a strong ability to generate syntactically valid CadQuery code. The 70% VSR proves that the combination of a Transformer architecture and a larger training dataset was effective at teaching the model the "grammar" of the CadQuery API.

Furthermore, the **Mean Best IoU of ~20%** indicates that the valid code produced is not random, but generates shapes that are geometrically and topologically similar to the ground truth. A qualitative review shows the model successfully identifies the primary features of a part (e.g., whether it's fundamentally a box or a cylinder) and generates the corresponding base code, even if it sometimes simplifies or omits more complex secondary operations. This result is a powerful proof-of-concept for the image-to-code approach.


## Bottlenecks & Limitations

- **Data Subsetting:** All results are based on training on only 25% of the available data due to the 7-hour time limit.
- **Character-Level Tokenizer:** While fast to implement, this tokenizer does not capture the semantic meaning of CadQuery functions (e.g., `Workplane` is seen as 9 independent characters).
- **CPU Bottleneck:** Even with pre-processed data files, on-the-fly data augmentation was a limiting factor for GPU utilization.

---

## Future Improvements (If More Time Available)

- **Full-Scale Training:** The most critical next step would be to train the final Transformer model on 100% of the dataset for 50-100 epochs to dramatically improve VSR and IoU.
- **Advanced Tokenizer:**  I would replace the character-level tokenizer with a **Byte-Pair Encoding (BPE) tokenizer**, trained on the CadQuery code corpus, to allow the model to learn and predict semantic sub-words.
- **Constrained Decoding:** I would enhance the Beam Search function to incorporate syntax rules (e.g., ensuring parentheses are always closed), which could push the VSR close to 100%.
- **GPU-Based Augmentation:** To eliminate the CPU bottleneck, I would integrate **NVIDIA DALI** to perform all data loading and augmentation directly on the GPU.
- **Hyperparameter Tuning:**  I would use a framework like **Optuna** to systematically search for the optimal learning rate, batch size, and Transformer architecture.

---

## Contact

For any questions or technical discussion:  
**Yacine Baghli** – yacine.baghli@gmail.com

---

> _This project demonstrates a principled and scalable approach to the image-to-CadQuery problem under tight time constraints._

