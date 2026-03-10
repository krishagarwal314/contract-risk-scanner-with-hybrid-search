import streamlit as st
import requests

backend_URL = "http://localhost:8000"

st.title("contract risk scanner")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None


uploaded_file = st.file_uploader("upload pdf", type="pdf")

if uploaded_file and st.session_state.session_id is None:
    with st.spinner("uploading doc..."):
        response = requests.post(
            f"{backend_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
        )
    data = response.json()

    if data.get("status") == "done":
        st.session_state.session_id = data["session_id"]
        st.success(f"uploaded, {data['chunks']} chunks indexed on pinecone")
    else:
        st.error("upload failed")

if st.session_state.session_id and st.session_state.scan_result is None:
    if st.button("scan the doc"):
        with st.spinner("scanning risk"):
            response = requests.post(
                f"{backend_URL}/scan",
                json={"session_id": st.session_state.session_id},
            )
            st.session_state.scan_result = response.json()
if st.session_state.scan_result:
    result = st.session_state.scan_result
    summary = result.get("summary", {})
    risks = result.get("risks", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("high", summary.get("high", 0))
    col2.metric("medium", summary.get("medium", 0))
    col3.metric("low", summary.get("low", 0))

    st.divider()
    st.subheader("risk breakdown")

    for key, data in risks.items():
        label = data.get("label", key)
        level = data.get("risk_level", "not_found")
        summary_text = data.get("summary", "")
        flag = data.get("flag")
        st.write(f"{label}: {level} — {summary_text} {flag or ''}")

    st.divider()