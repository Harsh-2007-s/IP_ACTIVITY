import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

st.set_page_config(layout="wide")

# =============================
# CSS → Fit Image to Screen
# =============================
st.markdown("""
<style>
img {
    max-height: 80vh;
    object-fit: contain;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Upload Image Screen
# =============================
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if st.session_state.uploaded_image is None:

    st.title("Image Processing Visualizer")

    uploaded_file = st.file_uploader(
        "Upload an Image to Begin",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.session_state.uploaded_image = image
        st.success("Image Uploaded Successfully!")

        st.rerun()

    st.stop()

# =============================
# Page Navigation
# =============================
if "page" not in st.session_state:
    st.session_state.page = 1

def next_page():
    if st.session_state.page < 6:
        st.session_state.page += 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

# =============================
# Use Uploaded Image
# =============================
image = st.session_state.uploaded_image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Make low contrast example
gray = cv2.normalize(gray, None, 40, 120, cv2.NORM_MINMAX)

# =============================
# Histogram Equalization
# =============================
equalized = cv2.equalizeHist(gray)

# =============================
# Bimodal Reference for Matching
# =============================
ref = np.zeros_like(gray)
ref[:, :gray.shape[1]//2] = 20
ref[:, gray.shape[1]//2:] = 230

matched = exposure.match_histograms(gray, ref).astype("uint8")

# =============================
# Pixel Matrix Helper
# =============================
def get_matrix(img):
    h, w = img.shape
    return img[h//2-4:h//2+4, w//2-4:w//2+4]

# =============================
# PAGE 1 – ORIGINAL
# =============================
if st.session_state.page == 1:

    st.title("Original Image")
    st.image(gray, use_container_width=True, clamp=True)

# =============================
# PAGE 2 – IMAGE DETAILS
# =============================
elif st.session_state.page == 2:

    st.title("Original Image Details")

    st.image(gray, use_container_width=True, clamp=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pixel Matrix (8x8)")
        st.write(get_matrix(gray))

    with col2:
        st.subheader("Histogram")
        fig, ax = plt.subplots()
        ax.hist(gray.ravel(),256,[0,256])
        st.pyplot(fig)

# =============================
# PAGE 3 – EQUALIZATION STEP
# =============================
elif st.session_state.page == 3:

    st.title("Histogram Equalization – Step by Step")

    if "eq_step" not in st.session_state:
        st.session_state.eq_step = 0
        st.session_state.eq_history = []

    steps = 15

    col1, col2 = st.columns([2,1])

    alpha = st.session_state.eq_step / steps
    blend = cv2.addWeighted(gray,1-alpha,equalized,alpha,0)

    col1.image(blend,use_container_width=True,clamp=True)

    col2.subheader("Pixel Matrix")
    col2.write(get_matrix(blend))

    if st.session_state.eq_step < steps:

        if st.button("Next Iteration ➡"):

            st.session_state.eq_history.append(
                (blend.copy(),get_matrix(blend))
            )

            st.session_state.eq_step += 1
            st.rerun()

    else:

        st.success("Equalization Completed")

        st.markdown("---")
        st.subheader("Transformation History")

        for idx,(img_iter,matrix_iter) in enumerate(st.session_state.eq_history):

            st.markdown(f"### Iteration {idx}")

            c1,c2 = st.columns([2,1])

            c1.image(img_iter,use_container_width=True,clamp=True)
            c2.write(matrix_iter)

# =============================
# PAGE 4 – MATCHING STEP
# =============================
elif st.session_state.page == 4:

    st.title("Histogram Matching – Step by Step")

    if "match_step" not in st.session_state:
        st.session_state.match_step = 0
        st.session_state.match_history = []

    steps = 15

    col1,col2 = st.columns([2,1])

    alpha = st.session_state.match_step / steps
    blend = cv2.addWeighted(gray,1-alpha,matched,alpha,0)

    col1.image(blend,use_container_width=True,clamp=True)

    col2.subheader("Pixel Matrix")
    col2.write(get_matrix(blend))

    if st.session_state.match_step < steps:

        if st.button("Next Iteration ➡"):

            st.session_state.match_history.append(
                (blend.copy(),get_matrix(blend))
            )

            st.session_state.match_step += 1
            st.rerun()

    else:

        st.success("Matching Completed")

        st.markdown("---")
        st.subheader("Transformation History")

        for idx,(img_iter,matrix_iter) in enumerate(st.session_state.match_history):

            st.markdown(f"### Iteration {idx}")

            c1,c2 = st.columns([2,1])

            c1.image(img_iter,use_container_width=True,clamp=True)
            c2.write(matrix_iter)

# =============================
# PAGE 5 – RESULTS
# =============================
elif st.session_state.page == 5:

    st.title("Results")

    st.subheader("Images")

    col1,col2,col3 = st.columns(3)

    col1.image(gray,use_container_width=True,clamp=True)
    col2.image(equalized,use_container_width=True,clamp=True)
    col3.image(matched,use_container_width=True,clamp=True)

    st.subheader("Pixel Matrices")

    col1,col2,col3 = st.columns(3)

    col1.write(get_matrix(gray))
    col2.write(get_matrix(equalized))
    col3.write(get_matrix(matched))

    st.subheader("Histograms")

    col1,col2,col3 = st.columns(3)

    fig1,ax1 = plt.subplots()
    ax1.hist(gray.ravel(),256,[0,256])
    col1.pyplot(fig1)

    fig2,ax2 = plt.subplots()
    ax2.hist(equalized.ravel(),256,[0,256])
    col2.pyplot(fig2)

    fig3,ax3 = plt.subplots()
    ax3.hist(matched.ravel(),256,[0,256])
    col3.pyplot(fig3)

# =============================
# PAGE 6 – FINAL COMPARISON
# =============================
elif st.session_state.page == 6:

    st.title("Final Comparison")

    col1,col2,col3 = st.columns(3)

    col1.image(gray,use_container_width=True,clamp=True)
    col2.image(equalized,use_container_width=True,clamp=True)
    col3.image(matched,use_container_width=True,clamp=True)

# =============================
# Navigation Buttons
# =============================
st.markdown("---")

col1,col2,col3 = st.columns([1,2,1])

col1.button("⬅ Previous",on_click=prev_page)
col3.button("Next ➡",on_click=next_page)
