# Third-party notices

The patches modify NVIDIA TensorRT-LLM source. Import hooks, benchmark helpers,
and frozen documentation-derived token fixtures are adapted from the original
TensorRT-LLM experiments. Original copyright and SPDX headers are retained.
The upstream license and bundled third-party notices are reproduced in LICENSE.

TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
FlashInfer: https://github.com/flashinfer-ai/flashinfer

The FlashInfer launch-cache control is an experimental common control applied
equally to both comparison arms. It is separate from the GDN replay patch.
