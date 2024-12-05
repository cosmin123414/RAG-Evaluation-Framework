import torch
from openai import OpenAI
import re
from typing import List, Dict, Callable

def split_into_sentences(text: str) -> List[str]:

    """
    Splits a text into sentence-long chunks.
    """

    # Use regular expressions
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    return sentence_endings.split(text.strip())



def extract_claims(text: str, openAI_client: OpenAI) -> List[str]:

    """
    Extract atomic claims from a given text using OpenAI API.

    """

    try:
      response = openAI_client.chat.completions.create(
          model="gpt-4o",
          messages=[
              {"role": "system", "content": "You return extracted claims separated by \\ characters, return no other text."},
              {
                  "role": "user",
                  "content": (
                      f"Extract all atomic claims from the following text:\n{text}"
                  ),
              },
          ],
          temperature=0.7,
      )

      raw_response = response.choices[0].message.content.strip()

      # Separate Claims
      claims = [claim.strip() for claim in raw_response.split("\\") if claim.strip()]
      return claims

    except Exception as e:
        print(f"Error extracting claims from sentence: {text}")
        print(f"Error message: {e}")
        return []



def process_text_to_claims(text: str, openAI_client: OpenAI) -> List[str]:

    """
    Splits text into sentences and extracts atomic claims from each sentence.

    Returns a list of claims
    """

    sentences = split_into_sentences(text)
    claims = []
    for sentence in sentences:
        claims.extend(extract_claims(sentence.strip(), openAI_client))

    return claims



def compute_relevant_count(answer_claims_embeddings: torch.Tensor, context_claims_embeddings: torch.Tensor, threshold: float) -> int:
    """
    Computes the number of claims in the answer that are relevant to the context.

    Relevance is determined if at least one context claim has a similarity
    above the threshold to a given answer claim.
    """
    relevant_count = 0

    for answer_embedding in answer_claims_embeddings:
        similarities = torch.nn.functional.cosine_similarity(answer_embedding.unsqueeze(0), context_claims_embeddings)
        if (similarities > threshold).any():
            relevant_count += 1

    return relevant_count



def negate_claims(claims: List[str], openAI_client: OpenAI) -> List[str]:
    """
     Negates each claim in a list of claims using openAI API.
    """
    negated_claims = []

    for claim in claims:

        try:
            response = openAI_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You negate claims, return only the negated claim, return no other text."},
                    {
                        "role": "user",
                        "content": f"Negate the following claim: '{claim}'",
                    },
                ],
                temperature=0.7,
            )
            negated_claim = response.choices[0].message.content.strip()
            negated_claims.append(negated_claim)

        except Exception as e:
            print(f"Error negating claim '{claim}': {e}")

    return negated_claims