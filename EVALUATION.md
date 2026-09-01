# Manga Translation Evaluation Framework

This document describes the evaluation system for assessing the quality of manga translations.

## Overview

The evaluation system includes:

1. **Single-Page Evaluation** (`evaluate_page.py`) - Evaluate a single translated page
2. **Batch Evaluation** (`batch_evaluate.py`) - Evaluate multiple pages from a book
3. **Batch Processing Pipeline** (`batch_pipeline.py`) - Process and evaluate multiple pages end-to-end

## Metrics

Currently uses **SacreBLEU** (BLEU metric from SacreBLEU package):
- Robust, standardized metric for translation evaluation
- Computes n-gram precision matches between hypothesis and reference
- Range: 0 (no matches) to 100 (perfect match)
- Note: BLEU is strict and requires exact word/n-gram matches

### Metric Output

Each evaluation returns:
- `scores`: List of per-sentence BLEU scores
- `mean`: Average BLEU score
- `std`: Standard deviation of scores
- `min/max`: Range of scores
- `corpus_bleu`: Overall BLEU score for the batch
- `corpus_precisions`: 1-gram, 2-gram, 3-gram, 4-gram precisions

## Usage

### 1. Single-Page Evaluation

Evaluate a single translated page against ground truth:

```bash
python src/evaluation/evaluate_page.py \
  --image "000.jpg" \
  --translation "outputs/translation.json" \
  --annotation "open-mantra-dataset/annotation.json"
```

**Output Example:**
```json
{
  "image": "000.jpg",
  "num_bubbles": 6,
  "num_references": 6,
  "scores": [0.0, 0.0],
  "mean": 0.0,
  "std": 0.0,
  "min": 0.0,
  "max": 0.0,
  "corpus_bleu": 0.0,
  "corpus_precisions": [0.0, 0.0, 0.0, 0.0]
}
```

### 2. Batch Evaluation

Evaluate multiple pages from a book:

```bash
python src/evaluation/batch_evaluate.py \
  --book "balloon_dream" \
  --annotation "open-mantra-dataset/annotation.json" \
  --translation-dir "batch_outputs/translations" \
  --max-pages 10 \
  --output "evaluation_results.json"
```

**Features:**
- Automatically discovers pages from annotation file
- Matches translation files to page images
- Aggregates statistics across all pages
- Saves detailed per-page results to JSON

**Output Structure:**
```json
{
  "book_name": "balloon_dream",
  "pages_evaluated": 10,
  "total_pages": 38,
  "stats": {
    "mean_bleu": 15.234,
    "std_bleu": 8.456,
    "min_bleu": 0.0,
    "max_bleu": 42.156,
    "median_bleu": 12.5
  },
  "page_results": [
    {
      "page_index": 0,
      "image": "000.jpg",
      "num_bubbles": 6,
      "mean": 0.0
    },
    ...
  ]
}
```

### 3. End-to-End Batch Processing

Process multiple pages through the complete pipeline:

```bash
python src/pipeline/batch_pipeline.py \
  --book "balloon_dream" \
  --annotation "open-mantra-dataset/annotation.json" \
  --images "open-mantra-dataset/images" \
  --model "manga_panel_detector_fp32.pt" \
  --output "batch_outputs" \
  --max-pages 5 \
  --conf 0.3
```

**Processing Stages:**
1. Detection + OCR + Context building + Translation
2. Inpainting (remove original text)
3. Typesetting (add translated text)

**Output Structure:**
```
batch_outputs/
├── translations/          # JSON files with translations
│   ├── 000_translation.json
│   ├── 001_translation.json
│   └── ...
├── inpainted/             # Images with text removed
├── typeset/               # Final typeset pages
└── stage1_*/stage2_*/     # Intermediate results
```

**After Batch Processing, Run Evaluation:**
```bash
python src/evaluation/batch_evaluate.py \
  --book "balloon_dream" \
  --translation-dir "batch_outputs/translations" \
  --annotation "open-mantra-dataset/annotation.json" \
  --output "evaluation_results.json"
```

## Annotation Format

The annotation.json file contains:

```json
[
  {
    "book_title": "balloon_dream",
    "pages": [
      {
        "page_index": 1,
        "image_paths": {"ja": "images/balloon_dream/ja/000.jpg"},
        "text": [
          {
            "x": 655, "y": 103, "w": 77, "h": 253,
            "text_ja": "夢の翼は",
            "text_en": "the wings of dreams,",
            "text_zh": "夢之翼"
          },
          ...
        ]
      },
      ...
    ]
  }
]
```

## Translation Format

The translation.json file from the pipeline contains:

```json
[
  {
    "index": 0,
    "bbox": [668.63, 109.05, 725.71, 349.84],
    "source_text": "夢の翼は",
    "translated_text": "The wings of dreams.",
    "scene_summary": "...",
    "crop_path": "outputs/contexts/text_0.png",
    "page_image": "open-mantra-dataset/images/balloon_dream/ja/000.jpg"
  },
  ...
]
```

## Improving Evaluation

### Current Limitations
- BLEU metric is strict and requires exact matches
- Low BLEU scores expected when translations are semantically correct but worded differently
- Better for comparing multiple systems than absolute quality assessment

### Recommended Next Steps

1. **Add Multiple Metrics:**
   - chrF (character-level F-score from SacreBLEU)
   - TER (Translation Edit Rate)
   - METEOR (semantic similarity)

2. **LLM-as-a-Judge:**
   - Use Ollama qwen2.5:7b to score translations on:
     - Fluency (readability in English)
     - Adequacy (correspondence to source)
     - Cultural appropriateness (for manga context)

3. **Manual Evaluation:**
   - Side-by-side comparison of source, reference, and hypothesis
   - Rating scales for different aspects

## Troubleshooting

### "No annotation found for image"
- Check image filename format (must be like "000.jpg")
- Verify image is in the correct book's section of annotation.json

### "No translation file found"
- Run the pipeline first to generate translation.json
- Check --translation-dir path matches where translations are saved

### BLEU score is very low (0.0 or near 0)
- This is expected if translations use different wording than references
- BLEU is very strict; consider using additional metrics
- Use batch_evaluate.py to see if it's consistent across pages

## Files

- `src/evaluation/bleurt_evaluator.py` - Evaluation metric wrapper
- `src/evaluation/evaluate_page.py` - Single-page evaluation CLI
- `src/evaluation/batch_evaluate.py` - Multi-page evaluation and statistics
- `src/pipeline/batch_pipeline.py` - End-to-end batch processing

## Requirements

- sacrebleu>=2.0.0 (in requirements.txt)
- Other dependencies from requirements.txt (ultralytics, PIL, manga-ocr, etc.)
