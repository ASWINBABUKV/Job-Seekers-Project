import joblib
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="IntelliHire",page_icon='images/favicon.png',layout="wide")

def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Use local CSS for mail
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css")

# for giff
lottie_coding = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_lqge6px5.json")
thank_mail=load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_zI9tjKnlQO.json")

#loading model
model = joblib.load('job_post_prediction_model.pkl')
vectorizer = TfidfVectorizer()


st.title(" IntelliHire")

# Input text box for required skills
required_skills = st.text_input("Enter required skills....")

if st.button("Check"):
    # Preprocess the input skills
    skills_list = [skill.strip() for skill in required_skills.split(",")]
    skills_text = " ".join(skills_list)
    skills_transformed = vectorizer.transform([skills_text])
    
    # Predict the job post
    predicted_job_post = model.predict(skills_transformed)[0]
    
    # Display the predicted job post
    st.success("Predicted Job Post: {}".format(predicted_job_post))
    

with st.container():
    st.write("---")
    left_column, right_column = st.columns((2,1))
    with left_column:
        st.header("What IntelliHire can do ?")
        st.write("##")
        st.write('<div style="text-align: justify;">This is a AI-based project for matching job seekers with suitable employment opportunities offers a powerful and efficient platform for job seekers and employers alike. Leveraging advanced artificial intelligence algorithms, this tool streamlines the job search process by analyzing and understanding the unique skills, qualifications, and preferences of job seekers. By employing sophisticated matching techniques, it intelligently matches job seekers with relevant job openings, increasing the likelihood of finding the perfect fit.</div>', unsafe_allow_html=True)
        st.write(
            """
            - AI-based solution provides a streamlined job search process by analyzing and understanding the unique skills, qualifications, and preferences of job seekers, saving them time and effort.
            - The AI-based solution bridges the gap between job seekers and suitable employment opportunities, empowering both candidates and employers with an intelligent platform that enhances their chances of finding the right talent and the perfect job match.
            - For employers, the AI tool simplifies and optimizes the recruitment process by identifying qualified candidates based on their skills, experience, and cultural fit, resulting in more successful hires.
            - The tool employs advanced algorithms to generate accurate and personalized recommendations, presenting job seekers with a curated list of employment opportunities that align with their skills and aspirations.
            """
        )
        # st.write("[YouTube Channel >](https://youtube.com/c/CodingIsFun)")
    with right_column:
        st_lottie(lottie_coding, height=500, key="jobseeking")   


with st.container():
    st.write("---")
    st.header("Get In Touch With Us!")
    st.write("##")
    contact_form = """
    <form action="https://formsubmit.co/parthasaradhih1@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns((1,2))
    with left_column:
        st_lottie(thank_mail, height=300, key="thank")
    with right_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    

    st.write("##")
    st.markdown("[Source Code of the project ... ](https://github.com/ASWINBABUKV/JOB_SEEKERS-PROJECT.git)")
    # st.markdown("[Github Repo...](https://github.com/ASWINBABUKV/JOB_SEEKERS-PROJECT.git)")

