"""
ai_engine.py — Offline AI engine for the SkyGuard ChatBot.
Uses a local HuggingFace model (no API calls, no internet needed after download).
Falls back to a smart rule-based engine if GPU/model is unavailable.
"""

import re
import warnings
warnings.filterwarnings("ignore")


class AIEngine:
    """Offline AI that answers questions about loaded datasets."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._embedder = None
        self._mode = "rule"  # "model" or "rule"
        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load a local text-generation model."""
        try:
            from transformers import pipeline
            # Use a small, fast model that runs on CPU
            self._model = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=-1,  # CPU
                max_new_tokens=256,
            )
            self._mode = "model"
            print("✅ AI Model loaded: google/flan-t5-base (offline, local)")
        except Exception as e:
            print(f"⚠️  AI model unavailable ({e}). Using smart rule-based engine.")
            self._mode = "rule"

        # Sentence embeddings for semantic search (optional)
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Semantic search loaded: all-MiniLM-L6-v2")
        except Exception:
            self._embedder = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer(self, question: str, context: str, df=None) -> str:
        """Answer a question given context about the dataset."""
        # First, try to handle with data-aware rules (fast, accurate)
        rule_answer = self._rule_based_answer(question, df)
        if rule_answer:
            return rule_answer

        # Fall back to model if available
        if self._mode == "model" and self._model:
            return self._model_answer(question, context)

        return self._fallback_answer(question, df)

    # ------------------------------------------------------------------
    # Model-based answer
    # ------------------------------------------------------------------
    def _model_answer(self, question: str, context: str) -> str:
        """Use the local transformer model to generate an answer."""
        # Truncate context to fit model input
        max_ctx = 1500
        if len(context) > max_ctx:
            context = context[:max_ctx] + "..."

        prompt = (
            f"Based on this dataset information, answer the question.\n\n"
            f"Dataset Info:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        try:
            result = self._model(prompt)[0]["generated_text"]
            return result.strip()
        except Exception as e:
            return f"Model error: {e}. Let me try the rule engine instead.\n\n" + \
                   self._fallback_answer(question, None)

    # ------------------------------------------------------------------
    # Smart rule-based engine (always works, no dependencies)
    # ------------------------------------------------------------------
    def _rule_based_answer(self, question: str, df=None) -> str | None:
        """Handle common data questions directly with pandas — fast & accurate."""
        if df is None:
            return None

        q = question.lower().strip()

        # --- Row / shape questions ---
        if any(kw in q for kw in ["how many rows", "how many records", "how many entries",
                                   "number of rows", "total rows", "count of rows", "row count",
                                   "how big", "dataset size"]):
            return f"The dataset has **{len(df):,}** rows and **{len(df.columns)}** columns."

        if any(kw in q for kw in ["how many columns", "number of columns", "column count",
                                   "what columns", "list columns", "show columns", "column names"]):
            cols = ", ".join(df.columns)
            return f"The dataset has **{len(df.columns)}** columns:\n`{cols}`"

        # --- Describe / summary ---
        if any(kw in q for kw in ["describe", "summary", "statistics", "stats",
                                   "overview", "summarize", "summarise"]):
            desc = df.describe(include="all").to_string()
            return f"📊 **Dataset Summary:**\n```\n{desc}\n```"

        # --- Missing values ---
        if any(kw in q for kw in ["missing", "null", "nan", "empty", "na values"]):
            missing = df.isnull().sum()
            total_missing = missing.sum()
            if total_missing == 0:
                return "✅ No missing values found in the dataset."
            lines = [f"**Missing values** (total: {total_missing:,}):"]
            for col, count in missing.items():
                if count > 0:
                    pct = count / len(df) * 100
                    lines.append(f"  - `{col}`: {count:,} ({pct:.1f}%)")
            return "\n".join(lines)

        # --- Data types ---
        if any(kw in q for kw in ["data type", "dtype", "types", "what type"]):
            dtypes = "\n".join(f"  - `{col}`: {dtype}" for col, dtype in df.dtypes.items())
            return f"**Column data types:**\n{dtypes}"

        # --- Head / sample ---
        if any(kw in q for kw in ["first", "head", "top rows", "sample", "show data",
                                   "preview", "show me", "example"]):
            n = 5
            match = re.search(r"(\d+)", q)
            if match:
                n = min(int(match.group(1)), 20)
            return f"**First {n} rows:**\n```\n{df.head(n).to_string()}\n```"

        # --- Tail ---
        if any(kw in q for kw in ["last", "tail", "bottom", "end rows"]):
            n = 5
            match = re.search(r"(\d+)", q)
            if match:
                n = min(int(match.group(1)), 20)
            return f"**Last {n} rows:**\n```\n{df.tail(n).to_string()}\n```"

        # --- Max / min for a column ---
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in q:
                if any(kw in q for kw in ["max", "maximum", "highest", "largest", "biggest"]):
                    if df[col].dtype in ("float64", "int64", "float32", "int32"):
                        val = df[col].max()
                        return f"The **maximum** value of `{col}` is **{val}**."
                if any(kw in q for kw in ["min", "minimum", "lowest", "smallest"]):
                    if df[col].dtype in ("float64", "int64", "float32", "int32"):
                        val = df[col].min()
                        return f"The **minimum** value of `{col}` is **{val}**."
                if any(kw in q for kw in ["mean", "average", "avg"]):
                    if df[col].dtype in ("float64", "int64", "float32", "int32"):
                        val = df[col].mean()
                        return f"The **average** value of `{col}` is **{val:.4f}**."
                if any(kw in q for kw in ["median"]):
                    if df[col].dtype in ("float64", "int64", "float32", "int32"):
                        val = df[col].median()
                        return f"The **median** value of `{col}` is **{val}**."
                if any(kw in q for kw in ["unique", "distinct", "categories"]):
                    n_unique = df[col].nunique()
                    if n_unique <= 20:
                        vals = ", ".join(str(v) for v in df[col].unique()[:20])
                        return f"`{col}` has **{n_unique}** unique values: {vals}"
                    return f"`{col}` has **{n_unique:,}** unique values."
                if any(kw in q for kw in ["distribution", "value counts", "count each",
                                          "how many of each", "frequency"]):
                    vc = df[col].value_counts().head(15).to_string()
                    return f"**Distribution of `{col}`:**\n```\n{vc}\n```"

        # --- Correlation ---
        if any(kw in q for kw in ["correlat", "relationship between", "related"]):
            try:
                numeric_df = df.select_dtypes(include=["number"])
                if len(numeric_df.columns) >= 2:
                    corr = numeric_df.corr().to_string()
                    return f"**Correlation matrix:**\n```\n{corr}\n```"
                return "Not enough numeric columns to compute correlation."
            except Exception:
                return "Could not compute correlations for this dataset."

        # --- Duplicates ---
        if any(kw in q for kw in ["duplicat"]):
            n_dup = df.duplicated().sum()
            if n_dup == 0:
                return "✅ No duplicate rows found."
            return f"Found **{n_dup:,}** duplicate rows ({n_dup / len(df) * 100:.1f}% of data)."

        # --- Shape ---
        if any(kw in q for kw in ["shape", "dimensions"]):
            return f"Dataset shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**"

        # --- Info ---
        if any(kw in q for kw in ["info", "metadata", "about this data"]):
            mem = df.memory_usage(deep=True).sum() / 1024 / 1024
            lines = [
                f"**Dataset Info:**",
                f"  - Rows: {len(df):,}",
                f"  - Columns: {len(df.columns)}",
                f"  - Memory: {mem:.2f} MB",
                f"  - Column names: {', '.join(df.columns)}",
            ]
            return "\n".join(lines)

        return None  # No rule matched — fall through to model

    def _fallback_answer(self, question: str, df=None) -> str:
        """Last-resort answer when no model and no rule matches."""
        suggestions = [
            "Try asking about: rows, columns, missing values, statistics, "
            "max/min/average of a column, correlations, duplicates, or data types.",
            "Example: 'How many rows?' or 'What is the average temperature?'"
        ]
        return ("I couldn't find a specific answer to that question.\n\n"
                + "\n".join(suggestions))
