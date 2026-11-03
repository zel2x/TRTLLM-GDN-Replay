# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Strict request-order audit; no runtime or numerical changes."""
import hashlib
import json
import random

def prompt_hash(prompt):
    return hashlib.sha256(json.dumps(prompt, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def describe(request):
    return {"prompt_sha256": prompt_hash(request.prompt), "input_len": request.prompt_len,
            "output_len": request.expected_output_len}

def expected_records(path, count):
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    random.Random(17).shuffle(entries)
    assert len(entries) == count
    return [{"prompt_sha256": prompt_hash(entry["input"]["messages"][1]["content"]),
             "input_len": entry["input"]["num_tokens"],
             "output_len": entry["input"]["max_tokens"]} for entry in entries]

def validate_sampled(sampled, expected):
    assert sampled == expected, "Actual sampled requests differ from pinned seeded fixture order"

def validate_result(result, sampled):
    count = len(sampled)
    assert result["completed"] == count and not any(result["errors"])
    assert result["input_lens"] == [row["input_len"] for row in sampled]
    assert result["output_lens"] == [row["output_len"] for row in sampled]
    assert result["total_output_tokens"] == sum(row["output_len"] for row in sampled)
