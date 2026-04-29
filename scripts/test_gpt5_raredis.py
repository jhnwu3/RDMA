#!/usr/bin/env python3
"""Quick smoke test for one RareDis sample with Azure GPT-5."""

import argparse
import pickle
import sys
import time
import traceback
import numpy as np
from pathlib import Path
from types import SimpleNamespace

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.rd.extractor import RDMAExtractor  # noqa: E402
from rdma.rd.verifier import RDMAVerifier  # noqa: E402
from rdma.utils.embedding import EmbeddingsManager  # noqa: E402
from rdma.utils.llm_client import AzureOpenAILLMClient  # noqa: E402
from rdma.utils.setup import setup_device  # noqa: E402

from datasets.raredis import RareDisDataset  # noqa: E402
from tasks.raredis import RareDisNER  # noqa: E402

_DEFAULT_EMBEDDINGS_FILE = (
    "/home/johnwu3/projects/rare_disease/workspace/repos/RDMA"
    "/data/vector_stores/rd_orpha_medembed.npy"
)
_DEFAULT_ABBREVIATIONS_FILE = (
    "/home/johnwu3/projects/rare_disease/workspace/repos/RDMA"
    "/data/tools/abbreviations_medembed_sm.npy"
)
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/raredis"


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test GPT-5 on one RareDis sample"
    )
    parser.add_argument("--model_type", type=str, default="gpt-5-john")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--condor", action="store_true")
    parser.add_argument(
        "--dataset_cache_dir", type=str, default=_DEFAULT_DATASET_CACHE_DIR
    )
    parser.add_argument(
        "--embeddings_file", type=Path, default=Path(_DEFAULT_EMBEDDINGS_FILE)
    )
    parser.add_argument(
        "--abbreviations_file", type=str, default=_DEFAULT_ABBREVIATIONS_FILE
    )
    parser.add_argument("--dotenv_path", type=str, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = SimpleNamespace(
        gpu_id=args.gpu_id,
        condor=args.condor,
        cpu=(args.gpu_id is None and not args.condor),
        retriever_gpu_id=None,
        retriever_cpu=False,
    )
    devices = setup_device(cfg)

    print("Loading RareDis sample...")
    dataset = RareDisDataset(num_workers=4, cache_dir=args.dataset_cache_dir)
    samples = dataset.set_task(RareDisNER(), num_workers=4)
    sample = next(iter(samples.subset(slice(args.sample_index, args.sample_index + 1))))

    doc_id = sample["patient_id"]
    text = pickle.loads(sample["text"])
    annotations = (
        pickle.loads(sample.get("annotations", b""))
        if sample.get("annotations")
        else []
    )
    print(f"Sample id: {doc_id}")
    print(f"Gold annotations: {len(annotations)}")
    if annotations:
        preview = annotations[:5]
        print(f"Gold preview: {preview}")

    llm_client = AzureOpenAILLMClient(
        model_type=args.model_type,
        azure_deployment=args.model_type,
        temperature=args.temperature,
        dotenv_path=args.dotenv_path,
    )

    embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
    embedding_manager = EmbeddingsManager(
        model_type="sentence_transformer",
        model_name="abhinand/MedEmbed-small-v0.1",
        device=devices.get("retriever", devices["llm"]),
    )

    extractor = RDMAExtractor(
        llm_client=llm_client,
        extraction_method="retrieval",
        embedding_manager=embedding_manager,
        embedded_documents=embedded_documents,
        window_size=5,
        top_k=5,
        min_sentence_size=50,
        strict=False,
        debug=args.debug,
    )
    verifier = RDMAVerifier(
        llm_client=llm_client,
        embedding_manager=embedding_manager,
        embedded_documents=embedded_documents,
        verifier_type="multi_stage",
        abbreviations_file=args.abbreviations_file,
        use_abbreviations=True,
        strict=False,
        exact_match=False,
        disease_check=False,
        debug=args.debug,
    )

    try:
        t0 = time.perf_counter()
        entities_with_contexts = extractor.extract_from_text(text)
        t1 = time.perf_counter()
        verified = verifier.verify_entities(entities_with_contexts)
        t2 = time.perf_counter()

        print("SUCCESS")
        print(f"Extracted entities: {len(entities_with_contexts)}")
        print(f"Verified entities:  {len(verified)}")
        print(f"Extraction time:    {t1 - t0:.2f}s")
        print(f"Verification time:  {t2 - t1:.2f}s")
        if entities_with_contexts:
            print("Extracted preview:")
            for item in entities_with_contexts[:5]:
                if isinstance(item, dict):
                    print(f"- {item.get('entity', '')}")
                else:
                    print(f"- {str(item)}")
        if not verified:
            print(
                "No entities survived verification on this sample. "
                "Try a different --sample_index or run with --debug "
                "for details."
            )
        print("Predicted preview:")
        for item in verified[:5]:
            print(f"- {item.get('entity', '')}")
    except Exception as exc:
        print(f"FAILED: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
