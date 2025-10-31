import os
import time
import json
import streamlit as st

from ticketbuddy.pipeline import TicketBuddyPipeline


@st.cache_resource(show_spinner=True)
def load_pipeline():
    return TicketBuddyPipeline.build()


def main():
    st.set_page_config(page_title="TicketBuddy RAG UI", page_icon="🛠️", layout="wide")
    st.title("TicketBuddy — IT Troubleshooting Assistant")
    st.caption("Enter an IT issue to retrieve relevant evidence and a structured fix.")

    with st.sidebar:
        st.subheader("Settings")
        use_openai = st.toggle("Use OpenAI for generation", value=bool(os.environ.get("OPENAI_API_KEY")))
        if use_openai and not os.environ.get("OPENAI_API_KEY"):
            key = st.text_input("OpenAI API Key", type="password")
            if key:
                os.environ["OPENAI_API_KEY"] = key
        st.markdown("---")
        st.write("Retrieval parameters are configured in `ticketbuddy/config.py`.")

    query = st.text_area("Your issue description", placeholder="Example: Outlook keeps prompting for password and VPN disconnects intermittently.", height=120)
    col1, col2 = st.columns([1, 1])
    with col1:
        run_btn = st.button("Get Answer", type="primary")
    with col2:
        show_raw = st.checkbox("Show raw JSON output")

    if run_btn and query.strip():
        with st.spinner("Building pipeline and retrieving evidence..."):
            t0 = time.time()
            pipeline = load_pipeline()
            # Show evidence for the full query
            contexts = pipeline.retrieve_contexts(query)
            # Generate final merged answer (handles multi-issue queries)
            answer = pipeline.answer(query)
            dt = time.time() - t0

        st.success(f"Completed in {dt:.2f}s")

        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Proposed Root Causes")
            rc = answer.get("root_causes", [])
            if rc:
                for item in rc:
                    st.markdown(f"- {item}")
            else:
                st.write("No root causes identified.")

            st.subheader("Resolution Steps")
            steps = answer.get("resolution_steps", [])
            if steps:
                for step in steps:
                    st.markdown(f"1. {step}")
            else:
                st.write("No resolution steps produced.")

        with right:
            st.subheader("Top Evidence")
            if contexts:
                for i, c in enumerate(contexts, 1):
                    src = c.get("source", "")
                    score = c.get("score", 0.0)
                    meta = c.get("source_meta", {})
                    with st.expander(f"{i}. {src}  (score: {score:.3f})"):
                        # Prefer a short preview
                        preview = (c.get("text", "") or "")
                        st.write(preview)
                        if meta:
                            st.caption(f"Metadata: {json.dumps(meta, ensure_ascii=False)}")
            else:
                st.write("No evidence found.")

        if show_raw:
            st.subheader("Raw JSON Output")
            st.code(json.dumps({"query": query, "answer": answer}, ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
