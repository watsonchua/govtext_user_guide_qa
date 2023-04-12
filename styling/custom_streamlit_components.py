import streamlit as st

"""
        <div class="info_banner_left_column">
            <img src="app/static/form_qr_code.png">
            <p class="image_caption">Scan for feedback form</p>
        </div>
        
        
                    Please help us improve by providing feedback 
            <a href="https://form.gov.sg/640ffbdbaed11b0011f4b4bb">here</a>
            . 
"""

def format_page_layout(page_title, layout="centered"):

    st.set_page_config(page_title, layout=layout)
    
    with open("styling/custom_styling.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def disclaimer_banner(container):

    html_text = """
    <div class="custom_info_banner">
        <div class="info_banner_right_column">
            <p style="color:red;">Attention:</p>
            This system is in alpha version.<br>
            While we strive to provide accurate and up-to-date information, there may be inaccuracies in the 
            content due to ongoing development and technical limitations.
        </div>
        <div class="info_banner_right_padding"></div>
    </div>
    """
    return container.markdown(html_text,unsafe_allow_html=True,)
