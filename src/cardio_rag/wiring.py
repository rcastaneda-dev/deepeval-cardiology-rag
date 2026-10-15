"""Create the real API clients and assemble the pipeline. Everything that needs keys lives here."""

from anthropic import Anthropic
from openai import OpenAI

from cardio_rag.config import CORPUS_DIR, INDEX_PATH, Settings, require_env
from cardio_rag.embeddings import make_embedder
from cardio_rag.index import Index, build_index, load_corpus, load_index, save_index
from cardio_rag.pipeline import RagPipeline
from cardio_rag.retriever import Retriever

_MAX_RETRIES = 5  # the SDKs retry rate limits and timeouts with exponential backoff


def _embedder(settings: Settings):
    client = OpenAI(api_key=require_env("OPENAI_API_KEY"), max_retries=_MAX_RETRIES)
    return make_embedder(client, settings.embedding_model)


def ingest(settings: Settings) -> Index:
    """Embed the whole corpus and write the index file."""
    index = build_index(load_corpus(CORPUS_DIR), _embedder(settings))
    save_index(index, INDEX_PATH)
    return index


def build_pipeline(settings: Settings, *, rebuild: bool = False) -> RagPipeline:
    embed = _embedder(settings)
    if rebuild or not INDEX_PATH.exists():
        index = ingest(settings)
    else:
        index = load_index(INDEX_PATH)
    client = Anthropic(api_key=require_env("ANTHROPIC_API_KEY"), max_retries=_MAX_RETRIES)
    return RagPipeline(Retriever(index, embed, settings.top_k), client, settings.generator_model)
