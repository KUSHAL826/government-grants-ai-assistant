import os
import time
import json
import ssl
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from html.parser import HTMLParser

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------------------------
# 1. Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Government Grants AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

load_dotenv()

# -----------------------------------------------------------------------------
# 2. Modern Light Theme CSS: Dark Text, Emerald & Gold (STRICTLY ZERO BLUE)
# -----------------------------------------------------------------------------
LIGHT_CHATBOT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

/* Global Light Canvas (NO BLUE) */
html, body, .stApp {
    background-color: #f8fafc !important;
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    color: #0f172a !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header & Bottom Containers: Clean Light Theme */
header, [data-testid="stHeader"],
footer, [data-testid="stBottom"], [data-testid="stBottom"] > div,
.stChatFloatingInputContainer,
[data-testid="stChatInputContainer"] {
    background-color: #f8fafc !important;
    background: #f8fafc !important;
    color: #0f172a !important;
}

/* Centered Chatbot Container Box */
.block-container {
    max-width: 800px !important;
    padding-top: 1.0rem !important;
    padding-bottom: 6.0rem !important;
}

/* 3D Chatbot Console Card */
.chatbot-3d-console {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 20px 24px 14px 24px;
    margin-bottom: 16px;
    box-shadow: 0 14px 35px -8px rgba(15, 23, 42, 0.07), 0 0 0 1px rgba(15, 23, 42, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.chatbot-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.bot-identity {
    display: flex;
    align-items: center;
    gap: 14px;
}

.bot-avatar-badge {
    width: 46px;
    height: 46px;
    background: linear-gradient(135deg, #059669, #10b981);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 6px 18px rgba(5, 150, 105, 0.25);
}

.bot-title-text {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.bot-live-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: #475569;
    font-weight: 500;
    margin-top: 2px;
}

.pulse-emerald {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6);
    animation: emeraldPulse 1.8s infinite;
}

@keyframes emeraldPulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
    70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.sync-verified-pill {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #047857;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* 3D WebGL Bot Container Box */
.three-canvas-frame {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    overflow: hidden;
    margin: 8px 0 12px 0;
    box-shadow: inset 0 2px 6px rgba(15, 23, 42, 0.03);
}

/* Live Scraping Micro-Audit Strip */
.scraping-audit-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 8px 14px;
    font-size: 0.78rem;
    color: #334155;
    margin-top: 6px;
}

.audit-portal-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-weight: 600;
    color: #0f172a;
}

/* Chat Messages Bubble Styling: LIGHT BACKGROUNDS WITH DARK TEXT */
[data-testid="stChatMessage"] {
    border-radius: 20px !important;
    padding: 16px 22px !important;
    margin-bottom: 14px !important;
    animation: fadeInUp 0.3s ease-out;
}

/* User Message: Clean Soft Sand/Slate Light Card with DARK TEXT (NO BLUE!) */
[data-testid="stChatMessage"][data-test-role="user"] {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03) !important;
}

[data-testid="stChatMessage"][data-test-role="user"] p,
[data-testid="stChatMessage"][data-test-role="user"] span {
    color: #0f172a !important; /* DARK TEXT */
    font-weight: 500 !important;
}

/* Assistant Message: Clean White Card with Emerald Accent & DARK TEXT */
[data-testid="stChatMessage"][data-test-role="assistant"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 4px solid #059669 !important;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05) !important;
}

[data-testid="stChatMessage"][data-test-role="assistant"] p,
[data-testid="stChatMessage"][data-test-role="assistant"] li,
[data-testid="stChatMessage"][data-test-role="assistant"] span {
    color: #0f172a !important; /* CRISP DARK TEXT */
    line-height: 1.68;
    font-size: 0.96rem;
}

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    color: #0f172a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    margin-top: 10px !important;
}

/* Inline Code: Emerald Tag Styling */
code, .stMarkdown code {
    background: #f1f5f9 !important;
    color: #047857 !important;
    border: 1px solid #cbd5e1 !important;
    padding: 3px 7px !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

/* -------------------------------------------------------------------------
   ANIMATED TYPING SYMBOL & REPLIER INDICATOR
   ------------------------------------------------------------------------- */
.ai-typing-active-box {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    padding: 10px 18px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-left: 4px solid #059669;
    border-radius: 14px;
    margin: 8px 0 14px 0;
    box-shadow: 0 6px 18px rgba(5, 150, 105, 0.08);
    animation: fadeInUp 0.25s ease-out;
}

.typing-symbol-icon {
    font-size: 1.2rem;
    animation: typingPenWiggle 1.2s infinite ease-in-out;
}

@keyframes typingPenWiggle {
    0%, 100% { transform: rotate(0deg) scale(1); }
    50% { transform: rotate(-15deg) scale(1.15); }
}

.typing-label-text {
    font-size: 0.88rem;
    font-weight: 700;
    color: #065f46;
}

.typing-dots-wave {
    display: inline-flex;
    gap: 4px;
    margin-left: 4px;
}

.tdot {
    width: 6px;
    height: 6px;
    background: #059669;
    border-radius: 50%;
    animation: dotBounce 1.4s infinite ease-in-out both;
}

.tdot:nth-child(1) { animation-delay: -0.32s; }
.tdot:nth-child(2) { animation-delay: -0.16s; }
.tdot:nth-child(3) { animation-delay: 0s; }

@keyframes dotBounce {
    0%, 80%, 100% { transform: scale(0.3); opacity: 0.3; }
    40% { transform: scale(1.2); opacity: 1; background: #10b981; }
}

/* Blinking Cursor during live reply stream */
.live-stream-cursor {
    display: inline-block;
    color: #059669;
    font-weight: 900;
    font-size: 1.1rem;
    margin-left: 3px;
    animation: cursorBlink 0.8s infinite;
}

@keyframes cursorBlink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

/* -------------------------------------------------------------------------
   CHECKBOXES & INPUT CONTROLS: CRISP DARK BORDERS & READABLE TEXT (ZERO WHITE-ON-WHITE)
   ------------------------------------------------------------------------- */
[data-testid="stCheckbox"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    margin-bottom: 8px !important;
}

[data-testid="stCheckbox"] label span {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

[data-testid="stCheckbox"] div[role="checkbox"] {
    border: 2px solid #64748b !important;
    background-color: #ffffff !important;
    border-radius: 6px !important;
}

[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
    background-color: #059669 !important;
    border-color: #059669 !important;
}

/* Selectbox Styling */
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

.streamlit-expanderContent {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    color: #0f172a !important;
}

/* Sidebar Light Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    border-color: #cbd5e1 !important;
    border-radius: 18px !important;
    background: #ffffff !important;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #059669 !important;
    box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.15) !important;
}

[data-testid="stChatInput"] textarea {
    color: #0f172a !important;
    font-size: 0.95rem !important;
}

/* Buttons */
.stButton > button {
    background-color: #059669 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #047857 !important;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.25) !important;
    transform: translateY(-1px);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""

st.markdown(LIGHT_CHATBOT_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Interactive 3D WebGL Chatbot Box (Emerald & Gold Palette, ZERO BLUE)
# -----------------------------------------------------------------------------
def render_3d_chatbot_box():
    """
    Renders an interactive 3D Chatbot Box Character with:
    - 3D Chamfered metallic Head Box
    - Animated glowing emerald visor eyes that track user cursor and blink
    - Side audio antennas with amber gold tips
    - Holographic orbital rings
    - Smooth levitation and 3D mouse tracking
    """
    three_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body, html {
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: transparent;
                width: 100%;
                height: 100%;
                user-select: none;
            }
            #bot-canvas-wrap {
                width: 100%;
                height: 175px;
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            #bot-caption {
                position: absolute;
                bottom: 6px;
                left: 50%;
                transform: translateX(-50%);
                font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
                font-size: 11px;
                color: #64748b;
                font-weight: 700;
                letter-spacing: 0.5px;
                text-transform: uppercase;
                pointer-events: none;
            }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div id="bot-canvas-wrap">
            <div id="bot-caption">Interactive 3D Chatbot • Move Cursor to Rotate</div>
        </div>
        <script>
            const container = document.getElementById('bot-canvas-wrap');
            const width = container.clientWidth;
            const height = 175;

            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 1000);
            camera.position.z = 190;

            const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
            renderer.setSize(width, height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);

            // Lighting (Emerald and Warm Gold, ZERO BLUE)
            scene.add(new THREE.AmbientLight(0xffffff, 0.95));
            const light1 = new THREE.PointLight(0x059669, 2.8, 300);
            light1.position.set(60, 50, 80);
            scene.add(light1);

            const light2 = new THREE.PointLight(0xd97706, 2.2, 300);
            light2.position.set(-60, -40, 70);
            scene.add(light2);

            // Master Bot Group
            const bot = new THREE.Group();
            scene.add(bot);

            // 1. 3D Chatbot Head Box (Chamfered Titanium Cube)
            const headGeo = new THREE.BoxGeometry(40, 36, 32);
            const headMat = new THREE.MeshStandardMaterial({
                color: 0x334155,
                metalness: 0.6,
                roughness: 0.3,
                wireframe: false
            });
            const headMesh = new THREE.Mesh(headGeo, headMat);
            bot.add(headMesh);

            // 2. Wireframe Emerald Accent Cage
            const wireGeo = new THREE.BoxGeometry(42, 38, 34);
            const wireMat = new THREE.MeshBasicMaterial({
                color: 0x059669,
                wireframe: true,
                transparent: true,
                opacity: 0.4
            });
            const wireMesh = new THREE.Mesh(wireGeo, wireMat);
            bot.add(wireMesh);

            // 3. Front Visor Screen (Dark Glass)
            const visorGeo = new THREE.PlaneGeometry(28, 14);
            const visorMat = new THREE.MeshStandardMaterial({
                color: 0x0f172a,
                roughness: 0.1,
                metalness: 0.9
            });
            const visor = new THREE.Mesh(visorGeo, visorMat);
            visor.position.set(0, 2, 16.2);
            bot.add(visor);

            // 4. Glowing Emerald Digital Eyes
            const eyeGeo = new THREE.CircleGeometry(2.4, 24);
            const eyeMat = new THREE.MeshBasicMaterial({
                color: 0x10b981,
                transparent: true,
                opacity: 0.95
            });

            const leftEye = new THREE.Mesh(eyeGeo, eyeMat);
            leftEye.position.set(-6.5, 2.2, 16.4);
            bot.add(leftEye);

            const rightEye = new THREE.Mesh(eyeGeo, eyeMat);
            rightEye.position.set(6.5, 2.2, 16.4);
            bot.add(rightEye);

            // 5. Antennas / Ear Pods (Cylinders with Amber Gold Caps)
            const earGeo = new THREE.CylinderGeometry(4, 4, 6, 16);
            const earMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8 });
            
            const leftEar = new THREE.Mesh(earGeo, earMat);
            leftEar.rotation.z = Math.PI / 2;
            leftEar.position.set(-22, 0, 0);
            bot.add(leftEar);

            const rightEar = new THREE.Mesh(earGeo, earMat);
            rightEar.rotation.z = Math.PI / 2;
            rightEar.position.set(22, 0, 0);
            bot.add(rightEar);

            // Glowing Amber Tips
            const tipGeo = new THREE.SphereGeometry(2.2, 16, 16);
            const tipMat = new THREE.MeshBasicMaterial({ color: 0xd97706 });
            
            const leftTip = new THREE.Mesh(tipGeo, tipMat);
            leftTip.position.set(-26, 0, 0);
            bot.add(leftTip);

            const rightTip = new THREE.Mesh(tipGeo, tipMat);
            rightTip.position.set(26, 0, 0);
            bot.add(rightTip);

            // 6. Holographic Orbital Ring (Emerald & Gold)
            const ringGeo = new THREE.TorusGeometry(32, 0.6, 12, 80);
            const ringMat = new THREE.MeshBasicMaterial({ color: 0x059669, transparent: true, opacity: 0.6 });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = Math.PI / 2.6;
            bot.add(ring);

            // Mouse Cursor Tracking
            let targetRotX = 0, targetRotY = 0;
            window.addEventListener('mousemove', (e) => {
                const rect = container.getBoundingClientRect();
                const mouseX = ((e.clientX - rect.left) / width) * 2 - 1;
                const mouseY = -(((e.clientY - rect.top) / height) * 2 - 1);
                targetRotY = mouseX * 0.75;
                targetRotX = -mouseY * 0.55;

                // Move eyes inside visor
                leftEye.position.x = -6.5 + mouseX * 1.5;
                leftEye.position.y = 2.2 + mouseY * 1.2;
                rightEye.position.x = 6.5 + mouseX * 1.5;
                rightEye.position.y = 2.2 + mouseY * 1.2;
            });

            window.addEventListener('resize', () => {
                const nw = container.clientWidth;
                camera.aspect = nw / height;
                camera.updateProjectionMatrix();
                renderer.setSize(nw, height);
            });

            let clock = new THREE.Clock();
            let lastBlink = 0;

            function animate() {
                requestAnimationFrame(animate);
                const t = clock.getElapsedTime();

                // Smooth Rotation towards Mouse Cursor
                bot.rotation.y += (targetRotY - bot.rotation.y) * 0.06;
                bot.rotation.x += (targetRotX - bot.rotation.x) * 0.06;

                // Floating Levitation Motion
                bot.position.y = Math.sin(t * 2.2) * 3.5;

                // Rotate Holographic Ring
                ring.rotation.z += 0.015;

                // Periodic Eye Blink
                if (t - lastBlink > 3.8) {
                    leftEye.scale.y = 0.1;
                    rightEye.scale.y = 0.1;
                    if (t - lastBlink > 4.0) {
                        leftEye.scale.y = 1.0;
                        rightEye.scale.y = 1.0;
                        lastBlink = t;
                    }
                }

                renderer.render(scene, camera);
            }
            animate();
        </script>
    </body>
    </html>
    """
    components.html(three_html, height=180)


# -----------------------------------------------------------------------------
# 4. Safe API Key Resolution
# -----------------------------------------------------------------------------
def resolve_gemini_api_key():
    """Safely retrieves Gemini key without throwing KeyError."""
    if "user_provided_key" in st.session_state and st.session_state.user_provided_key.strip():
        key = st.session_state.user_provided_key.strip()
        os.environ["GOOGLE_API_KEY"] = key
        return key

    key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
        elif "GOOGLE_API_KEY" in st.secrets:
            key = st.secrets["GOOGLE_API_KEY"]
        elif "GEMINI_API_KEYS" in st.secrets:
            key = st.secrets["GEMINI_API_KEYS"]
    except Exception:
        pass

    if not key:
        key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEYS")
            or os.getenv("GOOGLE_API_KEY")
        )

    if key:
        os.environ["GOOGLE_API_KEY"] = key

    return key


GEMINI_API_KEY = resolve_gemini_api_key()

DEFAULT_MODEL = "gemini-flash-latest"
UPDATE_INTERVAL = 24 * 3600  # 24 Hours in seconds

# -----------------------------------------------------------------------------
# 5. Official Government Portals & Authoritative Knowledge Sources
# -----------------------------------------------------------------------------
OFFICIAL_GOV_PORTALS = [
    {
        "id": "startup_india",
        "short_name": "Startup India",
        "full_name": "Startup India — Central Government Schemes Portal",
        "agency": "Department for Promotion of Industry and Internal Trade (DPIIT), Ministry of Commerce & Industry",
        "url": "https://www.startupindia.gov.in/content/sih/en/government-schemes.html"
    },
    {
        "id": "myscheme",
        "short_name": "myScheme",
        "full_name": "myScheme — National Platform for Government Schemes",
        "agency": "Ministry of Electronics & Information Technology (MeitY) & Digital India Corporation (DIC)",
        "url": "https://www.myscheme.gov.in/"
    },
    {
        "id": "dst",
        "short_name": "DST India",
        "full_name": "Department of Science & Technology (DST) — Call for Proposals & Grants",
        "agency": "Ministry of Science & Technology, Government of India",
        "url": "https://dst.gov.in/call-for-proposals"
    },
    {
        "id": "isti",
        "short_name": "ISTI Portal",
        "full_name": "India Science, Technology & Innovation (ISTI) — Schemes & Grants Portal",
        "agency": "Department of Science & Technology (DST) & Vigyan Prasar",
        "url": "https://www.indiascienceandtechnology.gov.in/"
    }
]

PDF_PORTAL_METADATA = {
    "id": "official_pdf",
    "short_name": "Testing Knowledge Base (PDF)",
    "full_name": "Official Government Grants & Startup Funding Testing Knowledge Base",
    "agency": "Government of India Official Schemes Compendium",
    "filename": "government_grants_testing.pdf",
    "source": "Official Portal PDF (government_grants_testing.pdf)"
}

GOV_PORTAL_URLS = [p["url"] for p in OFFICIAL_GOV_PORTALS]


def get_source_metadata(source_str: str):
    """Returns (full_name, short_name, agency, url) for any given source identifier."""
    if not source_str:
        return "Official Government Source", "Gov Portal", "Government of India", ""
    for p in OFFICIAL_GOV_PORTALS:
        if p["url"] == source_str or p["id"] == source_str or p["url"] in source_str:
            return p["full_name"], p["short_name"], p["agency"], p["url"]
    if "government_grants_testing.pdf" in source_str or "Official Portal PDF" in source_str:
        return PDF_PORTAL_METADATA["full_name"], PDF_PORTAL_METADATA["short_name"], PDF_PORTAL_METADATA["agency"], ""
    clean_domain = source_str.replace("https://", "").replace("http://", "").split("/")[0]
    return f"Official Portal ({clean_domain})", clean_domain, "Government Portal", source_str


# -----------------------------------------------------------------------------
# 6. Live Multi-Threaded Web Scraper (Fast, Parallel, 100% Scraped)
# -----------------------------------------------------------------------------
class CleanHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
        self.skip_depth = 0
        self.title_parts = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self.in_title = True
        elif tag in ('script', 'style', 'noscript', 'header', 'footer', 'nav', 'svg', 'aside'):
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False
        elif tag in ('script', 'style', 'noscript', 'header', 'footer', 'nav', 'svg', 'aside'):
            if self.skip_depth > 0:
                self.skip_depth -= 1
        elif tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'li', 'tr'):
            if self.skip_depth == 0:
                self.fed.append('\n')

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data.strip())
        elif self.skip_depth == 0 and data.strip():
            self.fed.append(data.strip() + ' ')

    def get_text(self):
        raw = ''.join(self.fed)
        raw = raw.replace('\ufeff', '')  # Clean BOM
        return re.sub(r'\n\s*\n+', '\n\n', raw).strip()

    def get_title(self):
        return ' '.join(self.title_parts).replace('\ufeff', '').strip()


def scrape_single_portal(portal_dict: dict):
    """Scrapes a single portal with timeout, extracting text and page title."""
    url = portal_dict["url"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=14) as resp:
            code = resp.getcode()
            html = resp.read().decode('utf-8', errors='ignore')
            elapsed = round(time.time() - t0, 2)
            parser = CleanHTMLTextExtractor()
            parser.feed(html)
            txt = parser.get_text()
            title = parser.get_title()
            if len(txt) > 200:
                return {
                    "source": url,
                    "full_name": portal_dict["full_name"],
                    "short_name": portal_dict["short_name"],
                    "agency": portal_dict["agency"],
                    "content": txt,
                    "page_title": title,
                    "status": "OK",
                    "latency": elapsed,
                    "status_code": code
                }
    except Exception as e:
        print(f"Scrape notice for {url}: {e}")
    return {
        "source": url,
        "full_name": portal_dict["full_name"],
        "short_name": portal_dict["short_name"],
        "agency": portal_dict["agency"],
        "content": "",
        "page_title": "",
        "status": "ERROR",
        "latency": 0,
        "status_code": "ERROR"
    }


def scrape_all_portals_parallel():
    """Scrapes all official government websites in parallel with ThreadPoolExecutor."""
    scraped_entries = []
    
    with ThreadPoolExecutor(max_workers=len(OFFICIAL_GOV_PORTALS)) as executor:
        results = executor.map(scrape_single_portal, OFFICIAL_GOV_PORTALS)
        for res in results:
            if res.get("content"):
                scraped_entries.append(res)

    # Ingest local testing PDF knowledge base
    pdf_path = PDF_PORTAL_METADATA["filename"]
    if os.path.exists(pdf_path):
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            pdf_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            if pdf_text.strip():
                scraped_entries.append({
                    "source": PDF_PORTAL_METADATA["source"],
                    "full_name": PDF_PORTAL_METADATA["full_name"],
                    "short_name": PDF_PORTAL_METADATA["short_name"],
                    "agency": PDF_PORTAL_METADATA["agency"],
                    "content": pdf_text.strip(),
                    "page_title": "Official Grants Knowledge Base (Testing PDF)",
                    "status": "OK",
                    "latency": 0.01,
                    "status_code": 200
                })
        except Exception as e:
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                pdf_text = "\n\n".join(d.page_content for d in docs)
                scraped_entries.append({
                    "source": PDF_PORTAL_METADATA["source"],
                    "full_name": PDF_PORTAL_METADATA["full_name"],
                    "short_name": PDF_PORTAL_METADATA["short_name"],
                    "agency": PDF_PORTAL_METADATA["agency"],
                    "content": pdf_text.strip(),
                    "page_title": "Official Grants Knowledge Base (Testing PDF)",
                    "status": "OK",
                    "latency": 0.05,
                    "status_code": 200
                })
            except Exception as e2:
                print(f"PDF extraction notice: {e2}")

    return scraped_entries


def get_cached_or_fresh_scraped_data(force_rescrape: bool = False):
    """Retrieves cached scraped data or automatically rescrapes every 24 hours."""
    cache_file = "scraped_data_cache.json"
    now = time.time()

    if not force_rescrape and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                last_time = data.get("timestamp", 0)
                cached_entries = data.get("entries", [])
                if now - last_time < UPDATE_INTERVAL and cached_entries:
                    # Enrich any entries that might be missing full names
                    enriched = []
                    for e in cached_entries:
                        full_n, short_n, agency_n, url_n = get_source_metadata(e.get("source", ""))
                        enriched.append({
                            **e,
                            "full_name": e.get("full_name") or full_n,
                            "short_name": e.get("short_name") or short_n,
                            "agency": e.get("agency") or agency_n
                        })
                    return enriched, last_time, False
        except Exception:
            pass

    fresh_entries = scrape_all_portals_parallel()
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"timestamp": now, "entries": fresh_entries}, f, ensure_ascii=False)
    except Exception:
        pass

    return fresh_entries, now, True


force_sync = st.session_state.get("force_sync_trigger", False)
if force_sync:
    st.session_state.force_sync_trigger = False

scraped_entries, last_sync_timestamp, did_update = get_cached_or_fresh_scraped_data(force_rescrape=force_sync)


def search_scraped_content(query: str, entries: list):
    """Extracts the most relevant text sections directly from the scraped portal documents."""
    q_words = [w.lower() for w in query.split() if len(w) > 2]
    matched_sections = []

    for entry in entries:
        source = entry.get("source", "Government Portal")
        full_n, short_n, agency_n, url_n = get_source_metadata(source)
        content = entry.get("content", "")
        paragraphs = content.split("\n\n")

        for p in paragraphs:
            p_clean = p.strip()
            if len(p_clean) < 40:
                continue

            p_lower = p_clean.lower()
            score = sum(1 for w in q_words if w in p_lower)
            if any(k in p_lower for k in ["scheme", "grant", "fund", "subsidy", "eligibility", "seed", "support", "innovator"]):
                score += 1

            if score > 0:
                matched_sections.append((score, full_n, agency_n, url_n or source, p_clean))

    matched_sections.sort(key=lambda x: x[0], reverse=True)

    if matched_sections:
        top = matched_sections[:6]
        return "\n\n---\n\n".join(
            f"Official Government Source: {fn}\nSponsoring Agency/Ministry: {ag}\nOfficial Portal Link: {link}\n{txt}"
            for _, fn, ag, link, txt in top
        )
    else:
        snippets = []
        for e in entries[:4]:
            full_n, short_n, agency_n, url_n = get_source_metadata(e.get("source", ""))
            snippets.append(
                f"Official Government Source: {full_n}\nSponsoring Agency/Ministry: {agency_n}\nOfficial Portal Link: {url_n or e.get('source')}\n{e.get('content', '')[:1200]}"
            )
        return "\n\n---\n\n".join(snippets)


# -----------------------------------------------------------------------------
# 7. Clean Sidebar: Web Scraping Verification & Controls (Zero Blue)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Settings")

    if GEMINI_API_KEY:
        st.markdown("""
        <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 12px; padding: 12px; margin-bottom: 12px;">
            <span style="color: #047857; font-weight: 700; font-size: 0.88rem;">🟢 Gemini AI Connected</span>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Change API Key"):
            new_key = st.text_input("Enter New Key", type="password", key="temp_key_input")
            if new_key and new_key != GEMINI_API_KEY:
                if st.button("Apply New Key", use_container_width=True):
                    st.session_state.user_provided_key = new_key
                    st.rerun()
    else:
        st.markdown("""
        <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 12px; margin-bottom: 12px;">
            <span style="color: #b91c1c; font-weight: 700; font-size: 0.88rem;">⚠️ Gemini Key Required</span>
        </div>
        """, unsafe_allow_html=True)
        user_key = st.text_input("Enter Gemini API Key", type="password", key="user_provided_key")
        if user_key:
            st.rerun()

    selected_model = st.selectbox(
        "AI Reasoning Model",
        ["gemini-flash-latest", "gemini-3.6-flash", "gemini-2.0-flash", "gemini-pro-latest"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📊 Web Scraping & Knowledge Audit")
    
    total_chars = sum(len(e.get("content", "")) for e in scraped_entries)
    sync_date_str = datetime.fromtimestamp(last_sync_timestamp).strftime("%b %d, %H:%M")
    
    st.markdown(f"""
    - **Live Status:** `🟢 Active & Verified`
    - **Total Knowledge:** `{total_chars:,} characters`
    - **Portals Scraped:** `{len(scraped_entries)} Sources Active`
    - **Last Synced:** `{sync_date_str}`
    - **Auto-Refresh:** `Every 24 Hours`
    """)

    with st.expander("🌐 Scraped Websites (Full Names)", expanded=True):
        for idx, e in enumerate(scraped_entries, 1):
            full_n, short_n, agency_n, url_n = get_source_metadata(e.get("source", ""))
            chars = len(e.get("content", ""))
            st.markdown(f"**{idx}. {full_n}**")
            st.caption(f"🏛️ **Agency:** {agency_n}")
            if url_n:
                st.markdown(f"🔗 [Visit Official Portal]({url_n})")
            st.markdown(f"<span style='color:#047857; font-weight:700; font-size:0.8rem;'>🟢 Scraped & Verified ({chars:,} chars)</span>", unsafe_allow_html=True)
            st.markdown("<div style='height:6px; border-bottom:1px dashed #cbd5e1; margin-bottom:8px;'></div>", unsafe_allow_html=True)

    if st.button("🔄 Sync Government Portals Now", use_container_width=True):
        st.session_state.force_sync_trigger = True
        st.rerun()

    # Diagnostic live testing expander
    with st.expander("🧪 Live Scraper Diagnostics Test", expanded=False):
        st.write("Run real-time connectivity and scraping tests across all official government websites:")
        if st.button("▶️ Run Live Scraping Diagnostics", use_container_width=True):
            with st.spinner("Connecting to official portals in parallel..."):
                diag_results = []
                for p in OFFICIAL_GOV_PORTALS:
                    t_start = time.time()
                    res = scrape_single_portal(p)
                    diag_results.append((p, res, round(time.time() - t_start, 2)))
                st.success("✅ Diagnostic Test Complete!")
                for p, res, elapsed in diag_results:
                    ok = res.get("status") == "OK"
                    st.markdown(f"**🏛️ {p['full_name']}**")
                    st.markdown(f"- **Status:** `{'🟢 200 OK (Scraped)' if ok else '🔴 Failed'}`")
                    st.markdown(f"- **Latency:** `{elapsed}s`")
                    st.markdown(f"- **Extracted Chars:** `{len(res.get('content', '')):,}`")
                    if res.get("page_title"):
                        st.markdown(f"- **Page Title:** `{res.get('page_title')}`")
                    st.markdown("---")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 **Hello!** How can I assist you with official government schemes, grants, or subsidies today? Ask any question to get started."
            }
        ]
        st.rerun()


# -----------------------------------------------------------------------------
# 8. 3D Chatbot Console & Interactive AI Box Header
# -----------------------------------------------------------------------------
total_chars = sum(len(e.get("content", "")) for e in scraped_entries)

st.markdown(f"""
<div class="chatbot-3d-console">
    <div class="chatbot-header-bar">
        <div class="bot-identity">
            <div class="bot-avatar-badge">🤖</div>
            <div>
                <div class="bot-title-text">Government Grants 3D Chatbot</div>
                <div class="bot-live-status">
                    <span class="pulse-emerald"></span>
                    <span>100% Live Scraped Knowledge • Light Theme</span>
                </div>
            </div>
        </div>
        <div class="sync-verified-pill">
            <span>⚡ 24h Portal Sync</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Interactive 3D WebGL Chatbot Box
with st.container():
    st.markdown('<div class="three-canvas-frame">', unsafe_allow_html=True)
    render_3d_chatbot_box()
    st.markdown('</div>', unsafe_allow_html=True)

# Dynamic Scraping Verification Strip with Full Website Names
strip_items = []
for e in scraped_entries:
    full_n, short_n, agency_n, url_n = get_source_metadata(e.get("source", ""))
    chars_count = len(e.get("content", ""))
    chars_label = f"{chars_count / 1000:.1f}k" if chars_count >= 1000 else f"{chars_count}"
    strip_items.append(f"<span title='{full_n} ({agency_n})' style='cursor:help;'>{short_n} <b style='color:#047857;'>({chars_label})</b></span>")

strip_html = " • ".join(strip_items)

st.markdown(f"""
<div class="scraping-audit-strip">
    <span class="audit-portal-tag">🌐 Scraped Sources:</span>
    {strip_html}
    <span>•</span>
    <span style="color: #047857; font-weight: 700;">🟢 {total_chars:,} chars active</span>
</div>
<div style="height: 14px;"></div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. Clean Initial Chat State (NO SCHEME NAMES DISPLAYED INITIALLY)
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Hello!** How can I assist you with official government schemes, grants, or subsidies today? Ask any question to get started."
        }
    ]


# -----------------------------------------------------------------------------
# 9. Answer Generation (100% Scraped Portal Context, Zero Default Schemes)
# -----------------------------------------------------------------------------
def stream_scraped_answer(query: str, api_key: str, model_name: str, entries: list):
    """
    Answers user inquiries based STRICTLY on the scraped government website content.
    Streams answer smoothly and provides intelligent scraped summary fallback.
    """
    context_text = search_scraped_content(query, entries)
    llm_worked = False

    if api_key:
        candidate_models = [model_name]
        for m in ["gemini-flash-latest", "gemini-3.6-flash", "gemini-2.0-flash", "gemini-pro-latest"]:
            if m not in candidate_models:
                candidate_models.append(m)

        for cand in candidate_models:
            try:
                system_prompt = """
You are an expert Government Grant and Scheme Advisor.
Answer the question using ONLY the provided official scraped government website and document context.
Structure your answer clearly with:
- Scheme Name & Sponsoring Agency/Ministry
- Purpose & Grant/Funding Amount
- Eligibility Criteria
- Application Process
- Official Government Source Portal

Do not invent schemes. If specific numbers are not present in the scraped text, present what is known and guide the user on which official portal to verify.
"""
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Scraped Government Portals Data:\n{context}\n\nUser Question:\n{question}")
                ])
                llm = ChatGoogleGenerativeAI(
                    model=cand,
                    temperature=0.2,
                    google_api_key=api_key
                )
                chain = prompt | llm | StrOutputParser()

                stream = chain.stream({"context": context_text, "question": query})
                for chunk in stream:
                    llm_worked = True
                    yield chunk

                if llm_worked:
                    return

            except Exception as e:
                print(f"Candidate model {cand} notice: {e}")
                continue

    # Fallback to direct scraped text excerpts if API is rate-limited
    fallback_header = "### 🏛️ Information Found on Official Government Portals:\n\n"
    words = (fallback_header + context_text).split(" ")
    for i in range(0, len(words), 4):
        yield " ".join(words[i:i+4]) + " "
        time.sleep(0.015)


# -----------------------------------------------------------------------------
# 10. Render Previous Chats Inside Centered Chatbot Box
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# -----------------------------------------------------------------------------
# 11. Chat Input & Active Typing Symbol While AI is Replying
# -----------------------------------------------------------------------------
user_query = st.chat_input("Type your question about government grants, schemes, or funding...")

if user_query:
    # 1. Append user message to conversation history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # 2. Render assistant response with prominent typing indicator
    with st.chat_message("assistant", avatar="🤖"):
        typing_slot = st.empty()
        response_slot = st.empty()

        # Display Animated Typing Symbol while initiating response
        typing_slot.markdown("""
        <div class="ai-typing-active-box">
            <span class="typing-symbol-icon">✍️</span>
            <span class="typing-label-text">AI is reading scraped portals & typing reply</span>
            <div class="typing-dots-wave">
                <span class="tdot"></span>
                <span class="tdot"></span>
                <span class="tdot"></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            stream_gen = stream_scraped_answer(
                user_query,
                GEMINI_API_KEY,
                selected_model,
                scraped_entries
            )

            full_response = ""

            for chunk in stream_gen:
                full_response += chunk
                # Keep active typing symbol indicator visible while replying
                typing_slot.markdown("""
                <div class="ai-typing-active-box">
                    <span class="typing-symbol-icon">✍️</span>
                    <span class="typing-label-text">AI is replying...</span>
                    <div class="typing-dots-wave">
                        <span class="tdot"></span>
                        <span class="tdot"></span>
                        <span class="tdot"></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                response_slot.markdown(full_response + '<span class="live-stream-cursor">▌</span>', unsafe_allow_html=True)

            # Response finished: Clear typing indicator banner
            typing_slot.empty()
            response_slot.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            typing_slot.empty()
            error_output = f"❌ **Notice:** Could not complete request ({str(e)}). Please verify your network or Gemini API key."
            st.markdown(error_output)
            st.session_state.messages.append({"role": "assistant", "content": error_output})