# Inference-only: qwen3_1_7b_scs_lora_prompt_scale_2_0 (test)
# RAG: BGE-M3 dense + BM25 + RRF + original BGE reranker.
# SCS-LoRA: PhoBERT predicted FULL probability vector is used only for soft expert routing.
# gold_test means the dataset split, never gold routing.
# Generator + adapter inference precision: FP16.

# %% [cell 1]
# Optional install command for notebook/Kaggle if packages are missing:
# %pip install -q transformers accelerate peft pandas numpy torch tqdm sentencepiece protobuf scikit-learn sentence-transformers faiss-cpu rank-bm25

# %% [cell 2]
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import inspect
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn

try:
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:
    faiss = None
    BM25Okapi = None
    SentenceTransformer = None
    CrossEncoder = None

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    set_peft_model_state_dict,
)

# %% [cell 3]
# ============================================================
# 1. FIXED EXPERIMENT CONFIG — MATCHES TRAINING
# ============================================================
SEED = 42
TYPE_ORDER = ['COMPARISON', 'FACTOID', 'SUMMARY', 'VERIFICATION']
TYPE2ID = {name: idx for idx, name in enumerate(TYPE_ORDER)}
ID2TYPE = {idx: name for idx, name in enumerate(TYPE_ORDER)}
NUM_EXPERTS = len(TYPE_ORDER)

TYPE_INSTRUCTIONS = {
    'COMPARISON': 'Trả lời bằng cách so sánh rõ ràng. Giữ đúng chiều tăng/giảm, cao hơn/thấp hơn và các số liệu liên quan.',
    'FACTOID': 'Trả lời ngắn gọn, trực tiếp. Giữ nguyên số liệu, đơn vị, tên công ty và thực thể tài chính trong ngữ cảnh.',
    'SUMMARY': 'Tóm tắt hoặc diễn giải ngắn gọn dựa trên ngữ cảnh. Không tự tạo thông tin ngoài ngữ cảnh.',
    'VERIFICATION': 'Trả lời theo hướng xác minh. Nếu phù hợp, bắt đầu bằng Đúng hoặc Không đúng, sau đó nêu bằng chứng ngắn gọn.',
}
PROMPT_SUFFIX = '\ntrả lời: '

BACKBONE_MODEL = 'HTThuanHcmus/qwen3-1.7b-merge'
TOKENIZER_MODEL = 'HTThuanHcmus/qwen3-1.7b-merge'
MODEL_FAMILY = 'qwen'
VARIANT_NAME = 'qwen3_1_7b_scs_lora_prompt_scale_2_0'
ADAPTER_FILENAME = 'qwen3_1_7b_scs_lora_prompt_scale_2_0_best_adapter.pt'
TRAINING_PROMPT_MODE = 'type_aware'
CASCADED_INPUT_SCALE = 2.0
EXPECTED_INJECTED_MODULES = 112

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

QUESTION_TYPE_CLASSIFIER_MODEL = 'tiam4tt/PhoBERT-VFQT-Cls'
CLASSIFIER_MAX_LEN = 256
CLASSIFIER_TEMPERATURE = 1.0

EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-m3')
RERANKER_MODEL = os.environ.get('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
RETRIEVAL_TOP_K = int(os.environ.get('RETRIEVAL_TOP_K', '20'))
RERANK_TOP_K = int(os.environ.get('RERANK_TOP_K', '3'))
RERANKER_MAX_LEN = int(os.environ.get('RERANKER_MAX_LEN', '1536'))
RERANK_BATCH_SIZE = int(os.environ.get('RERANK_BATCH_SIZE', '4'))
RETRIEVAL_EMBED_BATCH_SIZE = int(os.environ.get('RETRIEVAL_EMBED_BATCH_SIZE', '32'))

MAX_INPUT_LEN = int(os.environ.get('MAX_INPUT_LEN', '1500'))
GEN_MAX_NEW_TOKENS = int(os.environ.get('GEN_MAX_NEW_TOKENS', '256'))
GEN_NUM_BEAMS = int(os.environ.get('GEN_NUM_BEAMS', '4'))
INFER_BATCH_SIZE = max(1, int(os.environ.get('INFER_BATCH_SIZE', '4')))
RESUME_INFERENCE = os.environ.get('RESUME_INFERENCE', '1').strip().lower() not in {'0', 'false', 'no'}
MAX_INFERENCE_ROWS = int(os.environ.get('MAX_INFERENCE_ROWS', '0'))

INFERENCE_SPLIT = 'test'
if INFERENCE_SPLIT not in {'test', 'gold_test'}:
    raise ValueError(INFERENCE_SPLIT)

NUM_SHARDS = max(1, int(os.environ.get('NUM_SHARDS', '1')))
SHARD_ID = int(os.environ.get('SHARD_ID', '0'))
MERGE_SHARDS_ONLY = os.environ.get('MERGE_SHARDS_ONLY', '0').strip().lower() in {'1', 'true', 'yes'}
if not 0 <= SHARD_ID < NUM_SHARDS:
    raise ValueError(f'Invalid SHARD_ID={SHARD_ID} for NUM_SHARDS={NUM_SHARDS}')

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd().resolve()

OUTPUT_BASE_DIR = Path('/kaggle/working/inference_outputs')
OUTPUT_DIR = OUTPUT_BASE_DIR / VARIANT_NAME / INFERENCE_SPLIT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVAL_CACHE_DIR = Path(
    os.environ.get(
        'RETRIEVAL_CACHE_DIR',
        str(OUTPUT_BASE_DIR / '_retrieval_cache' / f'shard{SHARD_ID:02d}-of-{NUM_SHARDS:02d}'),
    )
).expanduser().resolve()

PRED_FINAL_PATH = OUTPUT_DIR / f'{VARIANT_NAME}_{INFERENCE_SPLIT}_predictions.csv'
if NUM_SHARDS == 1:
    PRED_PATH = PRED_FINAL_PATH
else:
    PRED_PATH = PRED_FINAL_PATH.with_name(
        f'{PRED_FINAL_PATH.stem}.shard{SHARD_ID:02d}-of-{NUM_SHARDS:02d}{PRED_FINAL_PATH.suffix}'
    )


INFERENCE_DTYPE = torch.float16

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# %% [cell 4]
# ============================================================
# 2. FIXED KAGGLE PATHS + UTILITIES
# ============================================================
def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    raise TypeError(type(obj).__name__)

DATA_DIR = Path('/kaggle/input/data-merge')
TRAIN_PATH = DATA_DIR / 'train.csv'
VAL_PATH = DATA_DIR / 'val.csv'
TEST_PATH = DATA_DIR / 'test.csv'
# Gold-test inference uses the manually annotated split.
GOLD_TEST_PATH = DATA_DIR / 'gold_test_annotated.csv'
INPUT_PATH = TEST_PATH if INFERENCE_SPLIT == 'test' else GOLD_TEST_PATH

ADAPTER_DIR = Path('/kaggle/input/merge-scs-lora-adapter')
ADAPTER_PATH = ADAPTER_DIR / ADAPTER_FILENAME

def resolve_adapter_path() -> Path:
    # Exact Kaggle adapter dataset path. No recursive auto-detection.
    if not ADAPTER_PATH.is_file():
        raise FileNotFoundError(
            f'Adapter not found: {ADAPTER_PATH}\n'
            f'Expected adapter dataset directory: {ADAPTER_DIR}'
        )
    return ADAPTER_PATH

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f'Missing Kaggle data directory: {DATA_DIR}')
if not INPUT_PATH.is_file():
    raise FileNotFoundError(f'Missing inference split: {INPUT_PATH}')
if not ADAPTER_DIR.is_dir():
    raise FileNotFoundError(f'Missing Kaggle adapter directory: {ADAPTER_DIR}')

def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return 'cuda out of memory' in msg or 'cuda error: out of memory' in msg


def _next_smaller_batch_size(current: int) -> int:
    if current <= 1:
        return 1
    if current <= 2:
        return 1
    return max(2, current // 2)


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print('VARIANT_NAME:', VARIANT_NAME)
print('INFERENCE_SPLIT:', INFERENCE_SPLIT)
print('TRAINING_PROMPT_MODE:', TRAINING_PROMPT_MODE)
print('CASCADED_INPUT_SCALE:', CASCADED_INPUT_SCALE)
print('ROUTING: PhoBERT predicted full probability soft routing')
print('DATA_DIR:', DATA_DIR)
print('INPUT_PATH:', INPUT_PATH)
print('ADAPTER_PATH:', ADAPTER_PATH)
print('PRED_PATH:', PRED_PATH)
print('FP16 inference:', INFERENCE_DTYPE)
# %% [cell 5]
# ============================================================
# 3. DATA + EXACT TRAIN/INFERENCE PROMPT FORMAT
# ============================================================
def canonical_question_type(value) -> str:
    # Exactly four labels are accepted. No aliases/fuzzy mapping.
    text = str(value).strip().upper()
    if text not in TYPE2ID:
        raise ValueError(
            f'Unknown question type {value!r}; expected exactly one of {TYPE_ORDER}'
        )
    return text

def optional_question_type(value) -> str:
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none'}:
        return ''
    return canonical_question_type(text)

def load_inference_csv(path: Path, require_gold: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {'question', 'context'}
    if require_gold:
        required.add('answer')
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns {sorted(missing)}; found {df.columns.tolist()}')
    for col in ['question', 'context', 'answer', 'question_type']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str).str.strip()
    df = df[df['question'].str.len().gt(0)].reset_index(drop=True)
    return df

def build_input(question: str, context: str, predicted_type: str) -> str:
    # Match SCS type-aware training text, using the predicted type at inference.
    qt = canonical_question_type(predicted_type)
    instruction = TYPE_INSTRUCTIONS[qt]
    return (
        f'loại câu hỏi: {qt}\n'
        f'hướng dẫn: {instruction}\n'
        f'câu hỏi: {str(question).strip()}\n'
        f'ngữ cảnh: {str(context).strip()}'
    )

def load_generator_tokenizer():
    attempts = [
        dict(use_fast=True, trust_remote_code=True, extra_special_tokens={}),
        dict(use_fast=True, trust_remote_code=True),
        dict(trust_remote_code=True, extra_special_tokens={}),
        dict(trust_remote_code=True),
        dict(use_fast=False, trust_remote_code=True, extra_special_tokens={}),
        dict(use_fast=False, trust_remote_code=True),
    ]
    last_error = None
    for kwargs in attempts:
        try:
            tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, **kwargs)
            if tok.eos_token_id is None:
                raise RuntimeError(f'Tokenizer {TOKENIZER_MODEL} has no eos_token_id')
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = 'left'
            print('Tokenizer class:', tok.__class__.__name__)
            print('Tokenizer is_fast:', bool(getattr(tok, 'is_fast', False)))
            print('Tokenizer load kwargs:', kwargs)
            return tok
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'Cannot load tokenizer {TOKENIZER_MODEL}: {last_error}')
# %% [cell 6]
# ============================================================
# 4. PHOBERT SOFT ROUTER (SCS ONLY)
# ============================================================
class QuestionTypeRouter:
    def __init__(self, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(QUESTION_TYPE_CLASSIFIER_MODEL)
        common = dict(low_cpu_mem_usage=True)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                QUESTION_TYPE_CLASSIFIER_MODEL,
                dtype=torch.float16 if str(device).startswith('cuda') else torch.float32,
                **common,
            )
        except TypeError:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                QUESTION_TYPE_CLASSIFIER_MODEL,
                torch_dtype=torch.float16 if str(device).startswith('cuda') else torch.float32,
                **common,
            )
        if str(device).startswith('cuda'):
            self.model = self.model.to(device=device, dtype=torch.float16)
        else:
            self.model = self.model.to(device=device)
        self.model.eval()

        raw = getattr(self.model.config, 'id2label', None) or {
            0: 'COMPARISON', 1: 'FACTOID', 2: 'SUMMARY', 3: 'VERIFICATION'
        }
        self.classifier_id2label = {
            int(k): canonical_question_type(v) for k, v in raw.items()
        }
        missing = set(TYPE_ORDER) - set(self.classifier_id2label.values())
        if missing:
            raise RuntimeError(
                f'Classifier labels do not cover adapter TYPE_ORDER. Missing={sorted(missing)}; '
                f'id2label={self.classifier_id2label}'
            )
        print('Classifier id2label:', self.classifier_id2label)

    def _remap(self, raw_probs: torch.Tensor) -> torch.Tensor:
        mapped = torch.zeros(
            raw_probs.shape[0], NUM_EXPERTS,
            device=raw_probs.device, dtype=raw_probs.dtype,
        )
        for classifier_id, label in self.classifier_id2label.items():
            mapped[:, TYPE2ID[label]] = raw_probs[:, classifier_id]
        sums = mapped.sum(dim=-1, keepdim=True)
        if torch.any(sums <= 0):
            raise RuntimeError('Invalid classifier probability mapping.')
        return mapped / sums

    @torch.inference_mode()
    def predict(self, questions: List[str]):
        enc = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=CLASSIFIER_MAX_LEN,
            return_tensors='pt',
        ).to(self.device)
        logits = self.model(**enc).logits
        if CLASSIFIER_TEMPERATURE != 1.0:
            logits = logits / CLASSIFIER_TEMPERATURE
        raw_probs = torch.softmax(logits, dim=-1)
        probs = self._remap(raw_probs)
        pred_ids = torch.argmax(probs, dim=-1).tolist()
        pred_types = [ID2TYPE[int(idx)] for idx in pred_ids]
        return pred_types, probs

# %% [cell 7]
# ============================================================
# 5. SCS-LORA INFERENCE ARCHITECTURE + COMPLETE ADAPTER LOAD
# ============================================================
class LoRAExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype, device=None):
        super().__init__()
        self.scaling = LORA_ALPHA / LORA_RANK
        self.dropout = nn.Dropout(LORA_DROPOUT)
        self.lora_A = nn.Linear(
            in_features, LORA_RANK, bias=False, dtype=dtype, device=device
        )
        self.lora_B = nn.Linear(
            LORA_RANK, out_features, bias=False, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter_dtype = self.lora_A.weight.dtype
        x_adapter = self.dropout(x.to(dtype=adapter_dtype))
        out = self.lora_B(self.lora_A(x_adapter)) * self.scaling
        return out.to(dtype=x.dtype)


class SCSLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.inference_route_weights: Optional[torch.Tensor] = None
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.experts = nn.ModuleList([
            LoRAExpert(self.in_features, self.out_features, dtype=dtype, device=device)
            for _ in range(NUM_EXPERTS)
        ])
        self.shared_out_expert = LoRAExpert(
            self.in_features, self.out_features, dtype=dtype, device=device
        )
        self.shared_input_expert = LoRAExpert(
            self.in_features, self.in_features, dtype=dtype, device=device
        )

    def set_inference_route_weights(self, weights: Optional[torch.Tensor]):
        self.inference_route_weights = weights

    def _mix_soft_experts(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        if alpha.shape[0] != x.shape[0]:
            if x.shape[0] % alpha.shape[0] != 0:
                raise RuntimeError(
                    f'Cannot expand routing batch {alpha.shape[0]} to hidden batch {x.shape[0]}'
                )
            # Beam search expands B -> B*num_beams.
            alpha = alpha.repeat_interleave(x.shape[0] // alpha.shape[0], dim=0)
        alpha = alpha.to(device=x.device, dtype=x.dtype)

        out = x.new_zeros((*x.shape[:-1], self.out_features))
        for expert_id, expert in enumerate(self.experts):
            expert_out = expert(x)
            weight = alpha[:, expert_id]
            while weight.dim() < expert_out.dim():
                weight = weight.unsqueeze(1)
            out = out + expert_out * weight.to(expert_out.dtype)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_route_weights is None:
            raise RuntimeError('SCS inference route weights are unset.')
        base_out = self.base_layer(x)
        shared_out = self.shared_out_expert(x)
        expert_input = x + CASCADED_INPUT_SCALE * self.shared_input_expert(x)
        type_out = self._mix_soft_experts(expert_input, self.inference_route_weights)
        return base_out + shared_out + type_out


def _get_parent_module(model: nn.Module, module_name: str):
    parts = module_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_scs_lora(model: nn.Module) -> nn.Module:
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.split('.')[-1] in TARGET_MODULES:
            parent, child_name = _get_parent_module(model, name)
            setattr(parent, child_name, SCSLoRALinear(module))
            replaced.append(name)
    print('Injected SCS-LoRA modules:', len(replaced))
    if len(replaced) != EXPECTED_INJECTED_MODULES:
        raise RuntimeError(
            f'Expected {EXPECTED_INJECTED_MODULES} q/k/v/o modules for {BACKBONE_MODEL}, '
            f'but injected {len(replaced)}.'
        )
    return model


def set_scs_route_weights(model: nn.Module, weights: Optional[torch.Tensor]):
    for module in model.modules():
        if isinstance(module, SCSLoRALinear):
            module.set_inference_route_weights(weights)


def _load_torch_payload(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_scs_adapter(model: nn.Module, path: Path) -> Dict:
    payload = _load_torch_payload(path)
    if not isinstance(payload, dict) or 'state_dict' not in payload:
        raise ValueError(f'Invalid SCS adapter payload: {path}')
    if payload.get('format') != 'scs_lora_adapter_v1':
        raise ValueError(
            f'Expected scs_lora_adapter_v1, got {payload.get("format")!r}'
        )

    meta = payload.get('meta', {})
    expected_prompt_meta = 'type_aware'
    expected = {
        'variant_name': VARIANT_NAME,
        'method': 'scs_lora',
        'backbone_model': BACKBONE_MODEL,
        'tokenizer_model': TOKENIZER_MODEL,
        'type_order': TYPE_ORDER,
        'training_routing': 'gold_one_hot',
        'training_prompt': expected_prompt_meta,
        'lora_rank': LORA_RANK,
        'lora_alpha': LORA_ALPHA,
        'lora_dropout': LORA_DROPOUT,
        'target_modules': TARGET_MODULES,
        'cascaded_input_scale': CASCADED_INPUT_SCALE,
        'shared_output_adapter': True,
        'shared_input_adapter': True,
    }
    for key, value in expected.items():
        if key not in meta:
            raise ValueError(f'Adapter metadata missing required field {key!r}')
        if meta[key] != value:
            raise ValueError(
                f'Adapter metadata mismatch {key}: expected {value!r}, got {meta[key]!r}'
            )

    state = payload['state_dict']
    state_numel = sum(t.numel() for t in state.values())
    if 'adapter_parameter_count' in meta and int(meta['adapter_parameter_count']) != state_numel:
        raise RuntimeError(
            f'Adapter parameter count mismatch: meta={meta["adapter_parameter_count"]}, state={state_numel}'
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected SCS keys: {unexpected[:20]}')
    print('Loaded complete SCS adapter:', path)
    print('SCS adapter tensors:', len(state), '| parameters:', f'{state_numel:,}')
    print('Missing base-model keys (expected):', len(missing))
    return meta

# %% [cell 9]
# ============================================================
# 7. FULL RAG RETRIEVER
# ============================================================
def _normalise_context(text: str) -> str:
    return ' '.join(str(text).strip().split())


def _rrf(rank_lists, k: int = 60):
    scores = defaultdict(float)
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked):
            scores[int(doc_id)] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class VietFinanceRetriever:
    def __init__(self, device: str):
        if any(x is None for x in [faiss, BM25Okapi, SentenceTransformer, CrossEncoder]):
            raise ImportError(
                'Install retrieval dependencies: sentence-transformers faiss-cpu rank-bm25'
            )
        self.device = device
        self.cache_dir = RETRIEVAL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_path = self.cache_dir / 'corpus.json'
        self.index_path = self.cache_dir / 'docs_bge_m3.faiss'
        self.corpus = self._load_or_build_corpus()
        print(f'Retrieval corpus: {len(self.corpus):,} unique contexts')

        print('Loading BGE-M3:', EMBEDDING_MODEL, '|', device)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        if str(device).startswith('cuda'):
            self.embedder = self.embedder.half()
        self.embedder.eval()
        self.faiss_index = self._load_or_build_faiss()

        self.bm25 = BM25Okapi([doc.lower().split() for doc in self.corpus])
        print('Loading original reranker:', RERANKER_MODEL, '|', device)
        self.reranker = CrossEncoder(
            RERANKER_MODEL,
            device=device,
            max_length=RERANKER_MAX_LEN,
        )

    def _load_or_build_corpus(self):
        if self.corpus_path.exists():
            with self.corpus_path.open('r', encoding='utf-8') as f:
                corpus = json.load(f)
            corpus = [x for x in corpus if _normalise_context(x)]
            if corpus:
                return corpus

        corpus, seen = [], set()
        for path in [TRAIN_PATH, VAL_PATH, TEST_PATH, GOLD_TEST_PATH]:
            if not path.exists():
                continue
            frame = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() == 'context')
            frame.columns = [str(c).strip().lower() for c in frame.columns]
            if 'context' not in frame.columns:
                continue
            for value in frame['context'].dropna().astype(str):
                context = _normalise_context(value)
                if context and context.lower() not in {'nan', 'none'} and context not in seen:
                    seen.add(context)
                    corpus.append(context)
        if not corpus:
            raise RuntimeError('Could not build retrieval corpus from train/val/test/gold_test contexts.')
        tmp = self.corpus_path.with_suffix('.json.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False)
        os.replace(tmp, self.corpus_path)
        return corpus

    def _load_or_build_faiss(self):
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            if index.ntotal == len(self.corpus):
                print('Loaded FAISS:', self.index_path)
                return index

        vectors = self.embedder.encode(
            self.corpus,
            batch_size=max(1, RETRIEVAL_EMBED_BATCH_SIZE),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype('float32')
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        tmp = self.index_path.with_suffix('.faiss.tmp')
        faiss.write_index(index, str(tmp))
        os.replace(tmp, self.index_path)
        return index

    def retrieve(self, question: str) -> Dict:
        q = str(question).strip()
        if not q:
            raise ValueError('Empty retrieval question.')

        dense_query = self.embedder.encode(
            [q], convert_to_numpy=True, normalize_embeddings=True
        ).astype('float32')
        dense_k = min(RETRIEVAL_TOP_K, len(self.corpus))
        _, dense_ids = self.faiss_index.search(dense_query, dense_k)
        dense_ranked = [int(i) for i in dense_ids[0] if int(i) >= 0]

        bm25_scores = np.asarray(self.bm25.get_scores(q.lower().split()))
        sparse_k = min(RETRIEVAL_TOP_K, len(self.corpus))
        sparse_ranked = np.argsort(-bm25_scores)[:sparse_k].astype(int).tolist()

        fused = _rrf([dense_ranked, sparse_ranked])[:RETRIEVAL_TOP_K]
        candidate_contexts = [self.corpus[i] for i in fused]
        if not candidate_contexts:
            raise RuntimeError('Retriever produced no candidates.')

        pairs = [[q, ctx] for ctx in candidate_contexts]
        scores = self.reranker.predict(
            pairs,
            batch_size=max(1, RERANK_BATCH_SIZE),
            show_progress_bar=False,
        )
        scores = np.asarray(scores, dtype=float).reshape(-1)
        order = np.argsort(-scores)
        reranked_contexts = [candidate_contexts[int(i)] for i in order]
        reranked_scores = [float(scores[int(i)]) for i in order]

        return {
            'context_used': reranked_contexts[0],
            'top_contexts': reranked_contexts[:RERANK_TOP_K],
            'top_scores': reranked_scores[:RERANK_TOP_K],
            'candidate_count': len(candidate_contexts),
            'reranked_contexts': reranked_contexts,
        }

# %% [cell 10]
# ============================================================
# 8. MODEL BUILD — BF16-SAVED ADAPTER -> FP16 INFERENCE
# ============================================================
def _load_causal_lm_fp16():
    common = dict(low_cpu_mem_usage=True, trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(
            BACKBONE_MODEL, dtype=INFERENCE_DTYPE, **common
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            BACKBONE_MODEL, torch_dtype=INFERENCE_DTYPE, **common
        )

def _adapter_dtypes(model: nn.Module):
    dtypes = set()
    for name, p in model.named_parameters():
        is_adapter = any(token in name for token in (
            '.experts.', '.shared_out_expert.', '.shared_input_expert.'
        ))
        if is_adapter:
            dtypes.add(str(p.dtype))
    return sorted(dtypes)

def build_inference_model(device: str):
    if not str(device).startswith('cuda'):
        raise RuntimeError('CUDA FP16 inference is required.')
    adapter_path = resolve_adapter_path()
    tokenizer = load_generator_tokenizer()
    model = _load_causal_lm_fp16()
    for p in model.parameters():
        p.requires_grad = False
    model = inject_scs_lora(model)
    meta = load_scs_adapter(model, adapter_path)
    model = model.to(device=device, dtype=INFERENCE_DTYPE)
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = True
    model.eval()
    dtypes = _adapter_dtypes(model)
    print('Inference adapter dtypes:', dtypes)
    if dtypes != ['torch.float16']:
        raise RuntimeError(f'Expected FP16 adapter parameters at inference, got {dtypes}')
    return model, tokenizer, meta, adapter_path

class SanitizeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

def clean_generation(text: str) -> str:
    text = str(text).strip()
    idx = text.lower().rfind('trả lời:')
    if idx >= 0:
        text = text[idx + len('trả lời:'):].strip()
    return text
# %% [cell 11]
# ============================================================
# 9. GENERATION
# ============================================================
@torch.inference_mode()
def generate_batch(model, tokenizer, router, retriever, rows: pd.DataFrame, device: str):
    questions = rows['question'].astype(str).tolist()
    retrieval_records = [retriever.retrieve(q) for q in questions]
    contexts_used = [r['context_used'] for r in retrieval_records]
    pred_types, probs = router.predict(questions)
    route_weights = probs.to(device=device, dtype=INFERENCE_DTYPE)

    prompts = [
        build_input(q, context_used, pred_type) + PROMPT_SUFFIX
        for q, context_used, pred_type in zip(questions, contexts_used, pred_types)
    ]
    inputs = tokenizer(
        prompts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_INPUT_LEN
    ).to(device)

    set_scs_route_weights(model, route_weights)
    try:
        generated = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            num_beams=GEN_NUM_BEAMS,
            do_sample=False,
            early_stopping=(GEN_NUM_BEAMS > 1),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([SanitizeLogitsProcessor()]),
        )
    finally:
        set_scs_route_weights(model, None)
    prompt_len = inputs['input_ids'].shape[1]
    predictions = []
    for seq in generated:
        decoded = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        predictions.append(clean_generation(decoded))
    return predictions, pred_types, probs.detach().float().cpu().numpy(), retrieval_records
# %% [cell 12]
# ============================================================
# 10. CRASH-SAFE RESUME + SHARDING
# ============================================================
def progress_path(csv_path: Path) -> Path:
    return Path(str(csv_path) + '.progress.jsonl')


def read_progress_rows(csv_path: Path) -> List[Dict]:
    p = progress_path(csv_path)
    rows = []
    if p.exists():
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and 'row_id' in rec:
                        rows.append(rec)
                except json.JSONDecodeError:
                    continue
    elif csv_path.exists():
        # Legacy/partial CSV resume: migrate existing completed rows into the
        # durable JSONL journal before appending new rows. Without this migration,
        # a later rebuild could omit rows that existed only in the old CSV.
        old_frame = pd.read_csv(csv_path, on_bad_lines='skip')
        if 'row_id' in old_frame.columns:
            rows = old_frame.to_dict('records')
            if rows:
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open('w', encoding='utf-8') as f:
                    for rec in rows:
                        f.write(json.dumps(rec, ensure_ascii=False, default=json_default) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
    dedup = {}
    for row in rows:
        try:
            dedup[int(row['row_id'])] = row
        except Exception:
            pass
    return [dedup[k] for k in sorted(dedup)]


def append_record(csv_path: Path, record: Dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    p = progress_path(csv_path)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_default) + '\n')
        f.flush()
        os.fsync(f.fileno())
    frame = pd.DataFrame([record])
    frame.to_csv(
        csv_path,
        mode='a', index=False,
        header=(not csv_path.exists() or csv_path.stat().st_size == 0),
        encoding='utf-8',
    )


def rebuild_sorted_csv(csv_path: Path) -> pd.DataFrame:
    rows = read_progress_rows(csv_path)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame['row_id'] = pd.to_numeric(frame['row_id'], errors='coerce')
    frame = frame.dropna(subset=['row_id'])
    frame['row_id'] = frame['row_id'].astype(int)
    frame = frame.sort_values('row_id').drop_duplicates('row_id', keep='last').reset_index(drop=True)
    tmp = Path(str(csv_path) + '.tmp')
    frame.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, csv_path)
    return frame


def predict_streaming(model, tokenizer, router, retriever, device: str):
    require_gold = INFERENCE_SPLIT == 'gold_test'
    df = load_inference_csv(INPUT_PATH, require_gold=require_gold)
    df['row_id'] = np.arange(len(df))
    if MAX_INFERENCE_ROWS > 0:
        df = df.head(MAX_INFERENCE_ROWS).copy()
    global_count = len(df)
    df = df[(df['row_id'] % NUM_SHARDS) == SHARD_ID].copy()
    print(f'Shard {SHARD_ID}/{NUM_SHARDS}: owns {len(df)} of {global_count} rows')

    if RESUME_INFERENCE and progress_path(PRED_PATH).exists():
        repaired = rebuild_sorted_csv(PRED_PATH)
        print('Rebuilt CSV from durable progress rows:', len(repaired))
    done = {int(r['row_id']) for r in read_progress_rows(PRED_PATH)} if RESUME_INFERENCE else set()
    pending = df[~df['row_id'].isin(done)].copy()
    print('Pending rows:', len(pending))

    cursor = 0
    current_batch = INFER_BATCH_SIZE
    bar = tqdm(total=len(pending), desc=f'{VARIANT_NAME} {INFERENCE_SPLIT}')
    while cursor < len(pending):
        bs = min(current_batch, len(pending) - cursor)
        batch = pending.iloc[cursor:cursor + bs].copy()
        try:
            preds, pred_types, probs, retrieval_records = generate_batch(
                model, tokenizer, router, retriever, batch, device
            )
        except Exception as exc:
            if _is_cuda_oom(exc) and bs > 1:
                current_batch = _next_smaller_batch_size(bs)
                print(f'CUDA OOM at batch={bs}; retrying same rows at batch={current_batch}')
                empty_cuda_cache()
                continue
            bar.close()
            raise

        for j, (_, row) in enumerate(batch.iterrows()):
            rr = retrieval_records[j]
            gold_context = _normalise_context(row.get('context', ''))
            reranked = [_normalise_context(x) for x in rr['reranked_contexts']]
            gold_rank = next((i + 1 for i, ctx in enumerate(reranked) if ctx == gold_context), -1) if gold_context else ''
            record = {
                'row_id': int(row['row_id']),
                'dataset': INFERENCE_SPLIT,
                'variant': VARIANT_NAME,
                'question': row.get('question', ''),
                'context': row.get('context', ''),
                'answer': row.get('answer', ''),
                'gold_question_type': optional_question_type(row.get('question_type', '')),
                'pred_question_type': pred_types[j],
                'cls_prob_comparison': float(probs[j][TYPE2ID['COMPARISON']]),
                'cls_prob_factoid': float(probs[j][TYPE2ID['FACTOID']]),
                'cls_prob_summary': float(probs[j][TYPE2ID['SUMMARY']]),
                'cls_prob_verification': float(probs[j][TYPE2ID['VERIFICATION']]),
                'prediction': preds[j],
                'context_used': rr['context_used'],
                'retrieved_contexts_top3': json.dumps(rr['top_contexts'], ensure_ascii=False),
                'reranker_scores_top3': json.dumps(rr['top_scores'], ensure_ascii=False),
                'retrieval_candidate_count': int(rr['candidate_count']),
                'gold_context_in_retrieved_top20': int(gold_context in set(reranked)) if gold_context else '',
                'gold_context_rank_after_rerank': gold_rank,
            }
            append_record(PRED_PATH, record)
            bar.update(1)
        cursor += len(batch)
    bar.close()
    final = rebuild_sorted_csv(PRED_PATH)
    print('Saved predictions:', PRED_PATH, '| rows:', len(final))
    return final


def sharded_path_for(shard_id: int) -> Path:
    if NUM_SHARDS == 1:
        return PRED_FINAL_PATH
    return PRED_FINAL_PATH.with_name(
        f'{PRED_FINAL_PATH.stem}.shard{shard_id:02d}-of-{NUM_SHARDS:02d}{PRED_FINAL_PATH.suffix}'
    )


def merge_shards():
    if NUM_SHARDS <= 1:
        if PRED_PATH.exists():
            return rebuild_sorted_csv(PRED_PATH)
        raise FileNotFoundError(PRED_PATH)
    all_rows = []
    for sid in range(NUM_SHARDS):
        path = sharded_path_for(sid)
        rows = read_progress_rows(path)
        print(f'Shard {sid}: {len(rows)} durable rows')
        all_rows.extend(rows)
    dedup = {int(r['row_id']): r for r in all_rows}
    ordered = [dedup[k] for k in sorted(dedup)]
    source_df = load_inference_csv(INPUT_PATH, require_gold=(INFERENCE_SPLIT == 'gold_test'))
    expected_count = min(len(source_df), MAX_INFERENCE_ROWS) if MAX_INFERENCE_ROWS > 0 else len(source_df)
    expected = set(range(expected_count))
    actual = set(dedup)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f'Incomplete merge: expected={len(expected)}, actual={len(actual)}, '
            f'missing(first20)={missing[:20]}, extra(first20)={extra[:20]}'
        )
    p = progress_path(PRED_FINAL_PATH)
    tmp_progress = Path(str(p) + '.tmp')
    with tmp_progress.open('w', encoding='utf-8') as f:
        for rec in ordered:
            f.write(json.dumps(rec, ensure_ascii=False, default=json_default) + '\n')
    os.replace(tmp_progress, p)
    frame = pd.DataFrame(ordered)
    tmp_csv = Path(str(PRED_FINAL_PATH) + '.tmp')
    frame.to_csv(tmp_csv, index=False, encoding='utf-8')
    os.replace(tmp_csv, PRED_FINAL_PATH)
    print('Merged verified predictions:', PRED_FINAL_PATH, '| rows:', len(frame))
    return frame

# %% [cell 13]
# ============================================================
# 11. MAIN
# ============================================================
def main():
    if MERGE_SHARDS_ONLY:
        return merge_shards()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. FP16 GPU inference is required.')
    device = 'cuda:0'
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.get_device_name(0))

    retriever = VietFinanceRetriever(device=device)
    model, tokenizer, adapter_meta, adapter_path = build_inference_model(device=device)
    print('Adapter path:', adapter_path)
    print('Adapter metadata:', adapter_meta)
    router = QuestionTypeRouter(device=device)
    try:
        result = predict_streaming(model, tokenizer, router, retriever, device)
    finally:
        del model
        del router
        del retriever
        empty_cuda_cache()
    return result

if __name__ == '__main__':
    result_df = main()
    if isinstance(result_df, pd.DataFrame):
        print(result_df.head(10).to_string(index=False))
