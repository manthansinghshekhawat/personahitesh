import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
API_VERSION = os.getenv("API_VERSION", "2025-01-01-preview")

# Page configuration
st.set_page_config(
    page_title="Hitesh Chaudhary AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .chat-message {
        background-color: #000;
        color: #fff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }
    .user-message { border-left: 4px solid #2196F3; }
    .assistant-message { border-left: 4px solid #9c27b0; }
    .stTextInput>div>div>input {
        background-color: #222 !important;
        color: #fff !important;
    }
    .stButton>button {
        background-color: #333 !important;
        color: #fff !important;
        border: none;
    }
    .stButton>button:hover {
        background-color: #444 !important;
    }
</style>
""", unsafe_allow_html=True)

class HiteshAIAssistant:
    def __init__(self):
        self.client = None
        self.conversation_history = []
        self._setup_client()

    def _setup_client(self):
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            st.error("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
            return
        try:
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=API_VERSION
            )
        except Exception as e:
            st.error(f"Error initializing client: {e}")

    def persona(self):
      return  '''
You are Hitesh Choudhary, an Electronics Engineer-turned educator with over 10 years of experience teaching programming and web development.
You founded LearnCodeOnline, served as CTO of iNeuron.ai, and currently work as Senior Director at PW (Physics Wallah). You manage two YouTube channels—one with over 1 million subscribers and another with 300,000—where you publish tutorials in both Hindi and English. You’re known for making complex topics simple, using real-world analogies, and encouraging hands-on practice. Here is how you teach:

**Teaching Style:**
- Explain each concept in simple language. Break down complex topics into small, logical steps.
- Use real-life analogies. For example, compare asynchronous calls in JavaScript to ordering tea with a token.
- Follow step-by-step progression: cover fundamentals (e.g., variables in Python) before advancing to intermediate topics (e.g., decorators in Python).
- Encourage learners to code along: “अब अपने कंप्यूटर पर VS Code खोलिए और नीचे के code टाइप कीजिए, फिर चलाइए और देखें output क्या आ रहा है।”
- Emphasize consistency: “रोज़ अगर ३० मिनट coding करेंगे तो एक महीने में फर्क दिखने लगेगा।”

**Personality & Communication:**
- Maintain an enthusiastic, energetic tone. Convey genuine excitement: “वाह, यह Trick बहुत काम आने वाला है!”
- Be patient and supportive: “अगर कोई गलती हो रही है, कोई बात नहीं—यही तो सीखने का तरीका है।”
- Use a mix of English and Hindi seamlessly. Start lessons with “चलिए शुरू करते हैं” and check understanding with “समझ गए?”
- Ask interactive questions: “क्या आपने यह तरीका पहले कभी इस्तेमाल किया है?”
- Provide positive reinforcement: “बहुत बढ़िया!” “शाबाश!”

**Expertise Areas:**
- Front-End: JavaScript (ES6+), React.js, Next.js, HTML/CSS, Tailwind CSS.
- Back-End: Node.js, Express.js, RESTful APIs, PostgreSQL, MongoDB.
- Full-Stack: End-to-end projects like a Dropbox-like Next.js app with Postgres, Clerk, and ImageKit; deployment on Vercel or AWS.
- Python: Basics to advanced, Django, Flask, scripting.
- DevOps: Docker containerization, CI/CD pipelines, AWS/DigitalOcean deployment.
- Interview Prep: Data structures, algorithms, problem solving (explained in bilingual sessions).
- Latest Tech Trends: New frameworks (e.g., Remix, Astro), open-source tools (tRPC, Supabase), best practices (microservices, serverless).

Always maintain Hitesh Choudhary’s style—explanations in simple language, practical examples, bilingual engagement, and motivational encouragement—when answering questions or teaching any topic.
'''

    def generate(self, msg):
        msgs = [{"role": "system", "content": self.persona()}] + self.conversation_history + [{"role": "user", "content": msg}]
        res = self.client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=msgs,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9
        )
        reply = res.choices[0].message.content
        # update history
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def is_end(self, text):
        return any(x in text.lower() for x in ['ok','thank you','thanks','bye','goodbye','धन्यवाद','ठीक है'])

    def save(self):
        st.session_state.setdefault('conversations', []).append({
            'timestamp': datetime.now().isoformat(),
            'messages': list(self.conversation_history)
        })

    def load(self):
        if 'current_conversation' in st.session_state:
            self.conversation_history = st.session_state.current_conversation

    def clear(self):
        self.conversation_history = []
        st.session_state.pop('current_conversation', None)

# Initialize assistant
if 'ai_assistant' not in st.session_state:
    st.session_state.ai_assistant = HiteshAIAssistant()
assistant = st.session_state.ai_assistant
assistant.load()

# Header
st.markdown("""
<div class="main-header">
  <h1>🎓 Hitesh Chaudhary AI Assistant</h1>
  <p>Personal Programming Mentor - चलिए कोडिंग सीखते हैं!</p>
</div>
""", unsafe_allow_html=True)

# Instructions and end button
st.info("Type 'thank you', 'ok', or click 🛑 End Conversation to finish and save chat.")
if st.button("🛑 End Conversation"):
    fb = "धन्यवाद! Happy coding! 🎉"
    assistant.conversation_history.append({"role":"assistant","content":fb})
    st.markdown(f"""
    <div class="assistant-message chat-message">
    <strong>Hitesh:</strong> {fb}
    </div>
    """, unsafe_allow_html=True)
    assistant.save()
    assistant.clear()
    st.stop()

# Capture user input
user_input = st.chat_input("Ask Hitesh anything about programming...")
if user_input:
    # Append user message
    assistant.conversation_history.append({"role": "user", "content": user_input})
    st.session_state.current_conversation = assistant.conversation_history
    # Display conversation so far (including user)
    for i in range(len(assistant.conversation_history)):
        msg = assistant.conversation_history[i]
        class_name = 'user-message' if msg['role']=='user' else 'assistant-message'
        st.markdown(f"""
        <div class=\"{class_name} chat-message\">{('<strong>You:</strong>' if msg['role']=='user' else '<strong>Hitesh:</strong>')} {msg['content']}
        </div>
        """, unsafe_allow_html=True)
    # Placeholder for typing animation
    typing_ph = st.empty()
    typing_ph.markdown("""
    <div class=\"assistant-message chat-message\">
       <strong>Hitesh:</strong> <em>typing...</em>
    </div>
    """, unsafe_allow_html=True)
    # Generate reply with spinner
    with st.spinner("Hitesh is thinking... 🤔"):
        time.sleep(0.5)
        reply = assistant.generate(user_input)
    # Replace typing placeholder with actual reply
    typing_ph.markdown(f"""
    <div class=\"assistant-message chat-message\">
       <strong>Hitesh:</strong> {reply}
    </div>
    """, unsafe_allow_html=True)
    st.session_state.current_conversation = assistant.conversation_history
    st.stop()

# If no new input, display existing history
if assistant.conversation_history:
    for i in range(len(assistant.conversation_history)):
        msg = assistant.conversation_history[i]
        class_name = 'user-message' if msg['role']=='user' else 'assistant-message'
        st.markdown(f"""
        <div class=\"{class_name} chat-message\">{('<strong>You:</strong>' if msg['role']=='user' else '<strong>Hitesh:</strong>')} {msg['content']}
        </div>
        """, unsafe_allow_html=True)
else:
    # Initial greeting
    greeting = "नमस्ते! I'm Hitesh Chaudhary, your programming mentor. चलिए शुरू करते हैं! 🚀"
    assistant.conversation_history.append({"role":"assistant","content":greeting})
    st.markdown(f"""
    <div class="assistant-message chat-message">
    <strong>Hitesh:</strong> {greeting}
    </div>
    """, unsafe_allow_html=True)
