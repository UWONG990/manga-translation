# Manga Translation Project

This repository is the active project for a manga translation pipeline built around visual context and open-source LMM evaluation.

## Project scope

The end-to-end workflow is:

1. Detect panels and text regions.
2. Extract OCR text from detected regions.
3. Build visual context from the page and bubble images.
4. Translate with a contextual model or LLM.
5. Typeset translated text back into the page.
6. Inpaint original text if a clean final page is needed.

## Reference repositories

The following repositories are kept as read-only references and are ignored by Git:

- Manga-Text-Segmentation
- multimodal-manga-translation
- open-mantra-dataset

They are used as benchmarks and implementation references, not as project source files.

## Current baseline

The first working module is a YOLO-based panel/text detector backed by the provided model.

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src\pipeline\run_detection.py --image "open-mantra-dataset\images\balloon_dream\ja\010.jpg"
```

## Gradio UI

Create a dedicated UI environment, because Gradio uses a newer Hugging Face
dependency range than the existing COMET evaluation environment:

```powershell
python -m venv venv-ui
.\venv-ui\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-ui.txt
python app.py
```

Run the pipeline from the same dedicated environment to avoid the protobuf
conflict in the existing COMET/BLEURT environment:

```powershell
.\venv-ui\Scripts\python.exe src\pipeline\prototype_pipeline.py `
  --image "D:\Project\manga-translation\open-mantra-dataset\images\tencho_isoro\ja\001.jpg" `
  --model "manga_panel_detector_fp32.pt"
```

The UI accepts a manga page, runs detection and OCR, summarizes the scene,
translates detected bubbles with Ollama, and can inpaint/typeset the result.

## Count translated bubbles

Count non-empty `text_en` bubble translations in `annotation.json`, grouped by
manga title/folder:

```powershell
python count_annotation_bubbles.py open-mantra-dataset\annotation.json
```

## Notes

- Keep the model under a local `models/` or project root directory.
- Use the reference repositories only as external references and not as editable source code.
