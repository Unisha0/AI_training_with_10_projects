import streamlit as st
from PIL import Image
from utils import preprocess_image, predict, load_random_sample, inject_custom_css
import time

# Inject chatbot-style CSS
inject_custom_css()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "start"
    st.session_state.image_tensor = None
    st.session_state.prediction = None
    st.session_state.image = None

st.markdown('<div class="chatbox">', unsafe_allow_html=True)

def chat_bubble(text, role="bot"):
    cls = "bot" if role == "bot" else "user"
    return f'<div class="chat-bubble {cls} clearfix">{text}</div>'

# Step 1: Greet user and ask input method
if st.session_state.page == "start":
    st.markdown(chat_bubble("👋 Hello! I’m your digit classifier bot. How would you like to input your digit?", "bot"), unsafe_allow_html=True)
    choice = st.radio("Choose input method:", ["🖌️ Draw Digit", "📤 Upload Image", "🎲 Random Sample"])
    if st.button("Continue"):
        st.session_state.choice = choice
        st.session_state.page = "input"

# Step 2: Get image input
elif st.session_state.page == "input":
    choice = st.session_state.choice

    if choice == "🖌️ Draw Digit":
        st.markdown(chat_bubble("✏️ Please draw your digit below.", "bot"), unsafe_allow_html=True)
        from streamlit_drawable_canvas import st_canvas
        canvas = st_canvas(
            fill_color="#000000",
            stroke_width=12,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )
        if canvas.image_data is not None:
            img = Image.fromarray((255 - canvas.image_data[:, :, 0]).astype('uint8'))
            st.image(img, caption="Drawn digit", width=150)
            if st.button("Submit Drawing"):
                st.session_state.image_tensor = preprocess_image(img)
                st.session_state.page = "predict"

    elif choice == "📤 Upload Image":
        st.markdown(chat_bubble("📁 Upload your image file (PNG/JPG)...", "bot"), unsafe_allow_html=True)
        file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])
        if file:
            image = Image.open(file)
            st.image(image, caption="Uploaded image", width=150)
            if st.button("Submit Upload"):
                st.session_state.image_tensor = preprocess_image(image)
                st.session_state.page = "predict"

    elif choice == "🎲 Random Sample":
        st.markdown(chat_bubble("🎲 Here's a random digit from MNIST.", "bot"), unsafe_allow_html=True)
        image, label = load_random_sample()
        st.image(image, caption=f"Label: {label}", width=150)
        if st.button("Use this sample"):
            st.session_state.image_tensor = preprocess_image(image)
            st.session_state.page = "predict"

# Step 3: Show prediction
elif st.session_state.page == "predict":
    st.markdown(chat_bubble("🔍 Processing your image...", "bot"), unsafe_allow_html=True)
    with st.spinner("Analyzing..."):
        time.sleep(1)
        prediction = predict(st.session_state.image_tensor)
        st.session_state.prediction = prediction

    st.markdown(chat_bubble(f"✅ I think it is: **{prediction}**", "bot"), unsafe_allow_html=True)

    # Retry options
    if st.button("🔁 Try Another"):
        st.session_state.page = "start"
        st.session_state.image_tensor = None
        st.session_state.prediction = None