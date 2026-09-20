"""Policy-aware RAG store used by the cross-generator DRPG variant (``rag_policy_rag``).

This is :class:`stream_bench.agents.utils.RAG` with one addition: every memory
entry also keeps the policy that was in force when the entry was written, so the
policy generator can retrieve past policies alongside past cases.
"""
import os
import random

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .utils import RetrieveOrder


class RAG:

    def __init__(self, rag_config: dict) -> None:
        model_path = rag_config["embedding_model"]
        if os.path.exists(model_path):
            # Load the embedding model from a local path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.embed_model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
        else:
            # Load the embedding model from the Hugging Face Hub
            self.tokenizer = AutoTokenizer.from_pretrained(rag_config["embedding_model"])
            self.embed_model = AutoModel.from_pretrained(rag_config["embedding_model"]).eval()
        print(f"Using embedding model: {rag_config['embedding_model']}")

        self.index = None
        self.id2evidence = dict()
        self.id2policy = dict()
        self.embed_dim = len(self.encode_data("Test embedding size"))
        self.insert_acc = 0

        self.seed = rag_config["seed"]
        self.top_k = rag_config["top_k"]
        orders = {member.value for member in RetrieveOrder}
        assert rag_config["order"] in orders
        self.retrieve_order = rag_config["order"]
        random.seed(self.seed)

        self.create_faiss_index()
        # TODO: make a file to save the inserted rows

    def create_faiss_index(self):
        # Create a FAISS index
        self.index = faiss.IndexFlatL2(self.embed_dim)

    def encode_data(self, sentence: str) -> np.ndarray:
        # Tokenize the sentence
        encoded_input = self.tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        feature = sentence_embeddings.numpy()[0]
        norm = np.linalg.norm(feature)
        if norm == 0 or np.isnan(norm):
            print(f"[Warning] norm=0 detected for sentence: {sentence!r}")
        return feature / norm

    def insert(self, key: str, value: str, policy: str = "") -> None:
        """Use the key text as the embedding for future retrieval of the value text, and store the policy."""
        embedding = self.encode_data(key).astype('float32')  # Ensure the data type is float32
        self.index.add(np.expand_dims(embedding, axis=0))
        self.id2evidence[str(self.insert_acc)] = value
        self.id2policy[str(self.insert_acc)] = policy
        self.insert_acc += 1

    def update(self, key: str, value: str, policy: str = "") -> None:
        """Update an existing entry in the RAG, including the policy."""
        embedding = self.encode_data(key).astype('float32')
        distances, indices = self.index.search(np.expand_dims(embedding, axis=0), 1)
        if len(indices[0]) > 0:
            idx = str(indices[0][0])
            if idx in self.id2evidence:
                self.id2evidence[idx] = value
                self.id2policy[idx] = policy
                print(f"Updated entry with key: {key[:50]}...")
                print(f"New Value: {value}")
                print(f"New Policy: {policy}")
            else:
                print(f"Warning: Entry not found for key: {key[:50]}...")
        else:
            print(f"Warning: No matching entry found for key: {key[:50]}...")

    def get_policy_by_index(self, idx: int) -> str:
        return self.id2policy.get(str(idx), "")

    def get_policy_by_key(self, key: str) -> str:
        embedding = self.encode_data(key).astype('float32')
        distances, indices = self.index.search(np.expand_dims(embedding, axis=0), 1)
        if len(indices[0]) > 0:
            idx = str(indices[0][0])
            return self.id2policy.get(idx, "")
        return ""

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Retrieve top-k text chunks"""
        embedding = self.encode_data(query).astype('float32')  # Ensure the data type is float32
        top_k = min(top_k, self.insert_acc)
        distances, indices = self.index.search(np.expand_dims(embedding, axis=0), top_k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        results = [{'link': str(idx), '_score': {'faiss': dist}} for dist, idx in zip(distances, indices)]
        # Re-order the sequence based on self.retrieve_order
        if self.retrieve_order == RetrieveOrder.SIMILAR_AT_BOTTOM.value:
            results = list(reversed(results))
        elif self.retrieve_order == RetrieveOrder.RANDOM.value:
            random.shuffle(results)

        text_list = [self.id2evidence[result["link"]] for result in results]
        return text_list

    def retrieve_by_correctness(self, query: str, top_k: int) -> tuple[list[str], list[str]]:
        """Retrieve top-k correct and incorrect text chunks separately.

        Args:
            query: The query text
            top_k: Number of examples to retrieve for each category

        Returns:
            A tuple of (correct_examples, incorrect_examples)
        """
        # First retrieve more examples than needed to ensure we have enough of each type
        embedding = self.encode_data(query).astype('float32')
        max_k = min(top_k * 10, self.insert_acc)  # Retrieve more to ensure we have enough of each type
        distances, indices = self.index.search(np.expand_dims(embedding, axis=0), max_k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        results = [{'link': str(idx), '_score': {'faiss': dist}} for dist, idx in zip(distances, indices)]

        # Re-order based on retrieve_order
        # if self.retrieve_order == RetrieveOrder.SIMILAR_AT_BOTTOM.value:
        #     results = list(reversed(results))
        # elif self.retrieve_order == RetrieveOrder.RANDOM.value:
        #     random.shuffle(results)

        # Separate correct and incorrect examples
        correct_examples = []
        incorrect_examples = []
        not_corr_verb = "not correct"

        for result in results:
            text = self.id2evidence[result["link"]]
            if text.strip().endswith(not_corr_verb):
                if len(incorrect_examples) < top_k:
                    incorrect_examples.append(text)
            else:
                if len(correct_examples) < top_k:
                    correct_examples.append(text)

            # Break if we have enough of both types
            if len(correct_examples) >= top_k and len(incorrect_examples) >= top_k:
                break

        return correct_examples, incorrect_examples

if __name__ == "__main__":
# Initialize RAG with a configuration dictionary
    rag_config = {
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "rag_filename": "test_rag_pool",
        "seed": 42,
        "top_k": 5,
        "order": "similar_at_top"  # ["similar_at_top", "similar_at_bottom", "random"]
    }
    rag = RAG(rag_config)

    # Key-value pairs for testing
    key_value_pairs = [
        ("Apple is my favorite fruit", "Oh really?"),
        ("What is your favorite fruit?", "Lettuce, tomato, and spinach."),
        ("What is your favorite vegetable?", "Apple, banana, and watermelon."),
        ("What do you like to read in your free time?", "Sherlock Holmes")
    ]

    # Insert the key-value pairs into the RAG
    for key, value in key_value_pairs:
        rag.insert(key, key + ' ' + value)

    from pprint import pprint

    query = "I like to eat lettuce."
    results = rag.retrieve(query, top_k=rag_config["top_k"])
    pprint(results)
