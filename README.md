# World Models

## Task 1: Evaluating Qwen2-VL-2B-Instruct on Multimodal Reasoning
- **Model and Dataset**: Used the Qwen2-VL-2B-Instruct model (from [GitHub](https://github.com/QwenLM/Qwen2-VL) and [Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)) and tested it on the MathVista dataset ([Hugging Face](https://huggingface.co/datasets/AI4Math/MathVista)).
- **Accelerated Inference**: Integrated the [vllm](https://github.com/vllm-project/vllm) toolkit to speed up inference processing.
- **Performance Evaluation**:
  - Assessed model performance on multimodal reasoning tasks.
  - Compared inference time with and without vllm.
  - Analyzed model predictions and identified bad cases.

## Task 2: Exploring Image Tokenization
- **VQGAN**: Reviewed the [VQGAN paper](https://compvis.github.io/taming-transformers/) and code ([GitHub](https://github.com/CompVis/taming-transformers)), focusing on the image tokenization process.
- **Model Execution**: Ran VQGAN on the DIV2K dataset ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)) using a subset of 200 images in [Google Colab](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb).
- **Visualizations**:
  - Visualized quantized token ID frequency distributions.
  - Performed dimensionality reduction (PCA and t-SNE) on token ID embeddings to study uniformity.
  - Compared quantized token differences between two similar images.

