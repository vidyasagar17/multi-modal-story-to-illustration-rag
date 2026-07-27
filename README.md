# Multi-Modal Story-to-Illustration Agent — Open-Source LLM + RAG Edition

Turn plain-text stories into fully illustrated storybooks, powered entirely by
**open-source language and image models** and grounded by a **Retrieval-Augmented
Generation (RAG)** layer for narrative and visual consistency.

## 🎯 Mission Statement

Our mission is to make illustrated storytelling **open, private, and consistent**.
We transform any written narrative into a coherent illustrated storybook using only
open-source models — no proprietary APIs, no per-image fees, and no data leaving the
user's machine. A retrieval-augmented memory keeps characters, settings, and art style
consistent from the first page to the last, so the result reads and looks like a single
cohesive book rather than a set of disconnected images. We aim to give writers,
educators, and researchers a transparent and reproducible pipeline they can inspect,
adapt, and run anywhere.

## 💡 Concept

This project is an open-source reimagining of the multi-modal story-to-illustration
pipeline. The original approach used GPT-4o-mini and DALL·E 2 to condense story pages
and generate art. This edition replaces those proprietary components with open-source
models and adds a RAG layer, so the whole system can run locally and stay consistent
across an entire story.

## 🔄 What's Different Here

| Stage | Original | This Project |
| --- | --- | --- |
| Language model | GPT-4o-mini | Open-source LLM (e.g. Llama 3 / Mistral / Qwen via Ollama or vLLM) |
| Image generation | DALL·E 2 | Open-source diffusion (e.g. Stable Diffusion / SDXL) |
| Consistency | Prompt-only | **RAG** over a character/style/scene memory |
| Hosting | Cloud API | Fully local / self-hosted |

## 🧱 Pipeline (planned)

1. **Text Condensation** — An open-source LLM distills each story page into a few
   vivid narrative sentences while preserving plot and characters.
2. **Retrieval (RAG)** — Before prompting the image model, the agent retrieves relevant
   context from a vector store (character sheets, established settings, prior scene
   descriptions, art-style notes) so new scenes stay consistent with earlier ones.
3. **Image Prompt Generation** — The LLM composes an image prompt that fuses the
   condensed page with the retrieved context.
4. **Image Generation** — An open-source diffusion model renders the illustration.
5. **Memory Update** — New characters, settings, and descriptions are embedded and
   written back into the vector store for future pages.

## 🛠️ Planned Tech Stack

- **LLM serving:** Ollama / vLLM (Llama 3, Mistral, or Qwen)
- **Image generation:** Stable Diffusion / SDXL (Diffusers)
- **Embeddings:** sentence-transformers
- **Vector store:** FAISS or ChromaDB
- **Orchestration:** agent framework (e.g. smolagents / LangGraph)

## 📊 Status

🚧 Early stage — this repository currently defines the concept, mission, and planned
architecture. Implementation is in progress.

## 📄 License

To be determined.
