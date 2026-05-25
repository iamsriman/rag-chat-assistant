def build_rag_prompt(context: str, history: str, question: str) -> str:
    # Prompt engineering note: context is placed before history/question so the
    # model treats retrieved evidence as the primary source of truth.
    return f"""You are a helpful assistant.

Use ONLY the provided context to answer.
Do not use outside knowledge.
If the answer is not present in the context, say exactly:
"I could not find enough information in the knowledge base to answer this question."

Context:
{context}

Conversation History:
{history}

Question:
{question}
"""
