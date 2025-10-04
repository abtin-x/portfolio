import streamlit as st
from huggingface_hub import InferenceClient
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("⚠️ Hugging Face token is not set. Please set HF_TOKEN environment variable.")

POPLER_BIN = r"C:\Users\abtin\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MODELS = {
    "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "Gemma 2B Instruct": "google/gemma-2b-it",
    "Zephyr 7B Beta": "HuggingFaceH4/zephyr-7b-beta"
}

MODEL_TASKS = {
    "mistralai/Mistral-7B-Instruct-v0.2": "chat",
    "HuggingFaceH4/zephyr-7b-beta": "chat",
    "google/gemma-2b-it": "text"
}

FALLBACK_MODELS = [
    ("google/gemma-2b-it", "text"),
    ("HuggingFaceH4/zephyr-7b-beta", "chat"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "chat"),
]

PERSONALITIES = {
    "Friendly Assistant": "You are a friendly and helpful AI assistant.",
    "Strict German Teacher": "You are a strict German teacher. Always reply in German, correct mistakes, and explain grammar rules.",
    "Funny Comedian": "You are a stand-up comedian. Always reply with humor and jokes.",
    "Coding Tutor": "You are a patient coding tutor. Always explain code step by step in detail."
}



def build_prompt(system_message: str, context: str, conversation):
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
    return f"{system_message}\n\nContext:\n{context}\n\n{chat_history}\nassistant:"


def chunk_text(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_faiss(all_chunks, embedder):
    texts = [c[1] for c in all_chunks]
    embeddings = embedder.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index


def retrieve_chunks(query, index, all_chunks, embedder, k=3):
    query_emb = embedder.encode([query])
    D, I = index.search(np.array(query_emb).astype("float32"), k)
    return [all_chunks[i] for i in I[0]]


st.set_page_config(page_title="Abtin's AI Assistant", page_icon="🤖")
st.title("🤖 Abtin's Multi-Modal AI Assistant")

tab1, tab2 = st.tabs(["💬 Chatbot", "🖼️ Image Generator"])

with tab1:
    selected_model_name = st.sidebar.selectbox("Choose a model", list(MODELS.keys()))
    selected_model = MODELS[selected_model_name]
    task = MODEL_TASKS.get(selected_model, "chat")

    selected_personality = st.sidebar.selectbox("Choose a personality", list(PERSONALITIES.keys()))
    system_message = PERSONALITIES[selected_personality]

    if "personality" not in st.session_state or st.session_state.personality != selected_personality:
        st.session_state.conversation = [{"role": "system", "content": system_message}]
        st.session_state.personality = selected_personality

    client = InferenceClient(model=selected_model, token=HF_TOKEN)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # File upload (for RAG)
    uploaded_files = st.sidebar.file_uploader("📂 Upload files", type=["txt", "pdf"], accept_multiple_files=True)
    all_chunks, index = [], None

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_content = ""

            if uploaded_file.type == "text/plain":
                file_content = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                try:
                    reader = PyPDF2.PdfReader(uploaded_file)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            file_content += text
                except:
                    pass

                # OCR fallback
                if len(file_content.strip()) < 20:
                    uploaded_file.seek(0)
                    images = convert_from_bytes(uploaded_file.read(), poppler_path=POPLER_BIN)
                    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
                    for img in images:
                        file_content += pytesseract.image_to_string(img)

            if file_content.strip():
                chunks = chunk_text(file_content)
                all_chunks.extend([(uploaded_file.name, chunk) for chunk in chunks])

        if all_chunks:
            index = build_faiss(all_chunks, embedder)
            st.sidebar.write(f"✅ Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")

    if st.sidebar.button("🧹 Clear Chat"):
        st.session_state.conversation = [{"role": "system", "content": system_message}]
        st.rerun()

    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})

        context = ""
        if index is not None:
            retrieved = retrieve_chunks(user_input, index, all_chunks, embedder, k=3)
            context = "\n\n".join([chunk for _, chunk in retrieved])
        else:
            retrieved = []

        prompt = build_prompt(system_message, context, st.session_state.conversation)

        # Try selected model, then fallback
        bot_text = None
        try:
            if task == "text":
                bot_text = client.text_generation(prompt, max_new_tokens=200)
            else:
                resp = client.chat_completion(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": f"{system_message}\n\nContext:\n{context}"},
                        *st.session_state.conversation
                    ],
                    max_tokens=200,
                    stream=False
                )
                bot_text = resp["choices"][0]["message"]["content"]
        except Exception as e:
            st.info(f"⚠️ {selected_model_name} failed. Trying fallback... ({e})")

        if not bot_text:
            for m_id, m_task in FALLBACK_MODELS:
                try:
                    backup_client = InferenceClient(model=m_id, token=HF_TOKEN)
                    if m_task == "text":
                        bot_text = backup_client.text_generation(prompt, max_new_tokens=200)
                    else:
                        resp = backup_client.chat_completion(
                            model=m_id,
                            messages=[
                                {"role": "system", "content": f"{system_message}\n\nContext:\n{context}"},
                                *st.session_state.conversation
                            ],
                            max_tokens=200,
                            stream=False
                        )
                        bot_text = resp["choices"][0]["message"]["content"]
                    st.success(f"✅ Answered using fallback: {m_id}")
                    break
                except Exception as e:
                    st.warning(f"⚠️ Fallback {m_id} also failed: {e}")
                    continue

        if not bot_text or bot_text.strip() == "":
            bot_text = "⚠️ Sorry, all models failed to respond. Please try again later."

        st.session_state.conversation.append({"role": "assistant", "content": bot_text})

        with st.chat_message("assistant"):
            st.write(bot_text)

        if retrieved:
            st.markdown("### 📚 Sources")
            for fname, _ in retrieved:
                st.markdown(f"- {fname}")

with tab2:
    st.header("🎨 AI Image Generator")
    img_client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN)

    if "images" not in st.session_state:
        st.session_state.images = []

    prompt_img = st.text_area("Enter your image prompt:")
    num_images = st.slider("Number of Images", 1, 4, 1)
    style = st.selectbox("Style Preset", ["None", "Realistic", "Cartoon", "Cyberpunk", "Oil Painting"])

    if st.button("Generate Image(s)"):
        if style != "None":
            prompt_img = f"{prompt_img}, in {style} style"
        with st.spinner("Generating image(s)..."):
            results = []
            for i in range(num_images):
                try:
                    image = img_client.text_to_image(prompt_img)
                    results.append(image)
                except Exception as e:
                    st.error(f"Image generation failed: {e}")
                    break
            for img in results:
                st.session_state.images.append(img)

    if st.session_state.images:
        st.subheader("🖼️ Generated Images Gallery")
        cols = st.columns(3)
        for i, img in enumerate(st.session_state.images):
            with cols[i % 3]:
                st.image(img, caption=f"Image {i + 1}", use_container_width=True)
                img.save(f"generated_{i + 1}.png")
                with open(f"generated_{i + 1}.png", "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download Image {i + 1}",
                        data=f,
                        file_name=f"generated_{i + 1}.png",
                        mime="image/png"
                    )
