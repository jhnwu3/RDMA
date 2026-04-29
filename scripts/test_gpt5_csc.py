#!/usr/bin/env python3
"""Quick smoke test for one CSC sample with Azure GPT-5."""

import argparse
import pickle
import sys
import time
import traceback
from pathlib import Path

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.hpo.extractor import PhenotypeExtractor  # noqa: E402
from rdma.hpo.verifier import HPOVerifier  # noqa: E402
from rdma.hpo.matcher import HPOMatcher  # noqa: E402
from rdma.utils.llm_client import AzureOpenAILLMClient  # noqa: E402

from datasets.csc import CSCDataset  # noqa: E402
from tasks.csc import CSCPhenotypeMining  # noqa: E402

_DEFAULT_EMBEDDINGS_FILE = (
    "/home/johnwu3/projects/rare_disease/workspace/repos/RDMA"
    "/data/vector_stores/G2GHPO_metadata_medembed.npy"
)
_DEFAULT_LAB_EMBEDDINGS_FILE = (
    "/home/johnwu3/projects/rare_disease/workspace/repos/RDMA"
    "/data/tools/lab_tables_medembed_sm.npy"
)
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/csc"


def main():
    parser = argparse.ArgumentParser(description="Smoke test GPT-5 on one CSC sample")
    parser.add_argument("--model_type", type=str, default="gpt-5-john")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--extraction_temperature", type=float, default=1.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--condor", action="store_true")
    parser.add_argument(
        "--dataset_cache_dir", type=str, default=_DEFAULT_DATASET_CACHE_DIR
    )
    parser.add_argument(
        "--embeddings_file", type=Path, default=Path(_DEFAULT_EMBEDDINGS_FILE)
    )
    parser.add_argument(
        "--lab_embeddings_file",
        type=Path,
        default=Path(_DEFAULT_LAB_EMBEDDINGS_FILE),
    )
    parser.add_argument("--dotenv_path", type=str, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("Loading CSC sample...")
    dataset = CSCDataset(cache_dir=args.dataset_cache_dir)
    samples = dataset.set_task(CSCPhenotypeMining())
    sample = next(iter(samples.subset(slice(args.sample_index, args.sample_index + 1))))

    doc_id = sample["patient_id"]
    text = pickle.loads(sample["text"])
    print(f"Sample id: {doc_id}")

    extraction_llm = AzureOpenAILLMClient(
        model_type=args.model_type,
        azure_deployment=args.model_type,
        temperature=args.extraction_temperature,
        dotenv_path=args.dotenv_path,
    )
    llm_client = AzureOpenAILLMClient(
        model_type=args.model_type,
        azure_deployment=args.model_type,
        temperature=args.temperature,
        dotenv_path=args.dotenv_path,
    )

    extractor = PhenotypeExtractor(
        llm_client=extraction_llm,
        extractor_type="retrieval",
        embeddings_file=str(args.embeddings_file),
        retriever="sentence_transformer",
        retriever_model="abhinand/MedEmbed-small-v0.1",
        top_k=5,
        negation=True,
        family_history=True,
        debug=args.debug,
    )
    verifier = HPOVerifier(
        llm_client=llm_client,
        embeddings_file=str(args.embeddings_file),
        verifier_version="v4",
        lab_embeddings_file=str(args.lab_embeddings_file),
        retriever="sentence_transformer",
        retriever_model="abhinand/MedEmbed-small-v0.1",
        debug=args.debug,
        use_demographics=True,
    )
    matcher = HPOMatcher(
        llm_client=llm_client,
        embeddings_file=str(args.embeddings_file),
        optimizer_version="standard",
        retriever="sentence_transformer",
        retriever_model="abhinand/MedEmbed-small-v0.1",
        top_k=5,
        debug=args.debug,
    )

    try:
        t0 = time.perf_counter()
        entities_with_contexts = extractor.extract([text])
        t1 = time.perf_counter()
        verified = verifier.verify(entities_with_contexts, text)
        t2 = time.perf_counter()
        matched = matcher.match(verified)
        t3 = time.perf_counter()

        print("SUCCESS")
        print(f"Extracted entities: {len(entities_with_contexts)}")
        print(f"Verified phenotypes: {len(verified)}")
        print(f"Matched phenotypes:  {len(matched)}")
        print(f"Extraction time:     {t1 - t0:.2f}s")
        print(f"Verification time:   {t2 - t1:.2f}s")
        print(f"Matching time:       {t3 - t2:.2f}s")
        print("Predicted preview:")
        for item in matched[:5]:
            print(f"- {item.get('hp_id', '')} | {item.get('term', '')}")
    except Exception as exc:
        print(f"FAILED: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
