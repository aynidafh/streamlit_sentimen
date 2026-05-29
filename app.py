import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.util import ngrams
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# PAGE CONFIG 
st.set_page_config(
    page_title="Analisis Sentimen Kasus Dugaan Korupsi Chromebook Nadiem Makarim dengan SVM",
    page_icon="🔍",
    layout="wide"
)

#  CUSTOM CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background utama */
.stApp {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c1c1c 0%, #111111 100%);
    border-right: 2px solid #FF6B00;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Judul sidebar */
[data-testid="stSidebar"] h1 {
    color: #FF6B00 !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
}

/* Radio button sidebar */
[data-testid="stSidebar"] .stRadio label {
    color: #cccccc !important;
    font-size: 0.95rem;
}

/* Header halaman */
h1 {
    color: #FF6B00 !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    border-bottom: 3px solid #FF6B00;
    padding-bottom: 0.5rem;
}

h2, h3 {
    color: #FF8C00 !important;
    font-weight: 700 !important;
}

/* Teks biasa */
p, li, label {
    color: #e0e0e0 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
    border: 1px solid #FF6B00;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 15px rgba(255, 107, 0, 0.2);
}

[data-testid="stMetricLabel"] {
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}

/* Tombol */
.stButton > button {
    background: linear-gradient(135deg, #FF6B00, #FF8C00) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(255, 107, 0, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(255, 107, 0, 0.6) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #1e1e1e !important;
    border: 1px solid #FF6B00 !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Text area */
.stTextArea > div > div > textarea {
    background-color: #1e1e1e !important;
    border: 1px solid #FF6B00 !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #1a1a1a;
    border-radius: 10px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    color: #cccccc !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #FF6B00, #FF8C00) !important;
    color: white !important;
}

/* Alert/info boxes */
.stSuccess {
    background-color: rgba(0, 200, 100, 0.15) !important;
    border-left: 4px solid #00c864 !important;
    border-radius: 8px !important;
}

.stError {
    background-color: rgba(255, 50, 50, 0.15) !important;
    border-left: 4px solid #ff3232 !important;
    border-radius: 8px !important;
}

.stInfo {
    background-color: rgba(255, 107, 0, 0.15) !important;
    border-left: 4px solid #FF6B00 !important;
    border-radius: 8px !important;
}

/* Slider */
.stSlider > div > div > div {
    background-color: #FF6B00 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #1e1e1e !important;
    border: 1px solid #FF6B00 !important;
    border-radius: 8px !important;
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

/* Divider */
hr {
    border-color: #FF6B00 !important;
    opacity: 0.3;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #FF6B00, #FF8C00) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #FF6B00; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# SLANG DICT 
slang_dict = {
    "sy":"saya","sya":"saya","aq":"saya","ak":"saya","gw":"saya","gue":"saya","gua":"saya","w":"saya",
    "km":"kamu","lo":"kamu","loe":"kamu","lu":"kamu",
    "gak":"tidak","ga":"tidak","gk":"tidak","ngga":"tidak","nggak":"tidak","ngak":"tidak",
    "kaga":"tidak","tdk":"tidak","tdak":"tidak","dak":"tidak","ndak":"tidak","ndk":"tidak",
    "gaada":"tidak ada","gada":"tidak ada","gabisa":"tidak bisa","gaboleh":"tidak boleh",
    "gamau":"tidak mau","gasuka":"tidak suka","mana peduli":"tidak peduli",
    "gapeduli":"tidak peduli","gajelas":"tidak jelas","gabener":"tidak benar",
    "ganyambung":"tidak nyambung","gasetuju":"tidak setuju","gaksetuju":"tidak setuju",
    "gaseharusnya":"tidak seharusnya","gatau":"tidak tahu","gtau":"tidak tahu",
    "gaktau":"tidak tahu","gausah":"tidak usah","jgn":"jangan",
    "udah":"sudah","udh":"sudah","uda":"sudah","sdh":"sudah","sudh":"sudah","sdah":"sudah",
    "blm":"belum","blom":"belum","belom":"belum",
    "yg":"yang","yng":"yang","dgn":"dengan","dg":"dengan",
    "krn":"karena","karna":"karena","krna":"karena","krena":"karena",
    "utk":"untuk","tp":"tapi","tpi":"tapi","jg":"juga","jga":"juga","sm":"sama",
    "dr":"dari","dri":"dari","drpd":"daripada","spt":"seperti","sprti":"seperti",
    "dlm":"dalam","pd":"pada","ttg":"tentang","bhw":"bahwa","sbg":"sebagai",
    "bgt":"banget","bgtt":"banget","bngt":"banget","bangett":"banget","banged":"banget","bnget":"banget",
    "sgt":"sangat","sngt":"sangat","lg":"lagi","lgi":"lagi","dl":"dulu","dlu":"dulu",
    "skrg":"sekarang","sblm":"sebelum","lgsg":"langsung","trs":"terus","trus":"terus",
    "ttp":"tetap","tetep":"tetap","sllu":"selalu","slu":"selalu","slalu":"selalu",
    "lbh":"lebih","lbih":"lebih","bnyk":"banyak","byk":"banyak","bnyak":"banyak",
    "dikit":"sedikit","sdikit":"sedikit","cpt":"cepat","cepet":"cepat",
    "smg":"semoga","smga":"semoga","smoga":"semoga","kmrn":"kemarin","kemaren":"kemarin",
    "mo":"mau","mw":"mau","cm":"cuma","pdhl":"padahal","padal":"padahal",
    "jd":"jadi","jdi":"jadi","bs":"bisa","bsa":"bisa",
    "dpt":"dapat","dpet":"dapat","dapet":"dapat","pke":"pakai","pkai":"pakai","pake":"pakai",
    "blg":"bilang","nemu":"menemukan","dtg":"datang","dateng":"datang",
    "nyampe":"sampai","sampe":"sampai","sampek":"sampai","smpe":"sampai","ampe":"sampai",
    "hrs":"harus","hrus":"harus","msh":"masih","psti":"pasti","bkn":"bukan",
    "tau":"tahu","ksh":"kasih","liat":"lihat","ngeliat":"melihat",
    "mikir":"berpikir","males":"malas","cape":"capek","maksa":"memaksa","denger":"dengar",
    "ngajak":"mengajak","ngerti":"mengerti","ngulang":"mengulang","ngikutin":"mengikuti",
    "ngeluh":"mengeluh","ngomong":"berbicara","ngerasa":"merasa","ngelibatin":"melibatkan",
    "nguntungin":"menguntungkan","ngebela":"membela","ngebahas":"membahas",
    "ngebuktiin":"membuktikan","ngeributin":"memperdebatkan",
    "dibelain":"dibela","belain":"bela","nyuruh":"menyuruh","nyari":"mencari",
    "nyetor":"menyetor","nyentuh":"menyentuh","nyorot":"menyoroti",
    "korup":"korupsi","bag":"bagian",
    "nih":"ini","ni":"ini","nii":"ini","tuh":"itu","tu":"itu",
    "gini":"begini","ginii":"begini","gtu":"begitu","gt":"begitu","gitu":"begitu","gituu":"begitu",
    "gmna":"bagaimana","gimana":"bagaimana","gimane":"bagaimana",
    "klo":"kalau","kalo":"kalau","kl":"kalau","klw":"kalau",
    "ky":"kayak","kyk":"kayak","kek":"kayak",
    "gampang":"mudah","rapih":"rapi","mantep":"mantap",
    "bner":"benar","bnr":"benar","bnrr":"benar","bener":"benar",
    "murce":"murah","baguss":"bagus","sukaa":"suka",
    "wkwk":"tertawa","wkwkwk":"tertawa","hehe":"tertawa","jir":"kaget","anjir":"kaget",
    "cuy":"teman","cok":"teman","org":"orang","orng":"orang",
    "emg":"memang","emang":"memang","aslii":"asli","rame":"ramai",
    "ogah":"tidak mau","ngawur":"tidak masuk akal","ngaco":"kacau",
    "makin":"semakin","taun":"tahun","blio":"beliau","aja":"saja","doang":"saja","amp":"sampai",
    "scr":"secara","qrt":"quote retweet","sosmed":"media sosial","buzzer":"pendengung",
    "netizen":"warganet","ojol":"ojek online","menfess":"mention confess","techbro":"pengusaha teknologi",
    "uu":"undang undang","tipikor":"tindak pidana korupsi","ikn":"ibu kota nusantara",
    "apbn":"anggaran pendapatan dan belanja negara","pns":"pegawai negeri sipil",
    "rp":"rupiah","jt":"juta","m":"miliar","t":"triliun",
    "mksh":"terima kasih","makasih":"terima kasih","makasi":"terima kasih",
    "tq":"terima kasih","tks":"terima kasih","thx":"terima kasih","maci":"terima kasih",
}

custom_stopwords = [
    'yang','dan','di','ke','dari','ini','itu','dengan','untuk',
    'adalah','atau','juga','dalam','sudah','saya','aku','ges','ku',
    'kami','kita','mereka','dia','ia','anda','akan','dapat',
    'pada','oleh','karena','namun','tetapi','tapi','jika','kalau',
    'maka','agar','mau','bagi','nya','pun','lebih','sangat','sekali',
    'hanya','saja','ya','lagi','masih','harus','boleh','memang',
    'terus','lalu','setelah','sebelum','saat','ketika','pernah',
    'cara','hal','banyak','beberapa','semua','setiap',
    'tersebut','seperti','bahwa','tolong','mohon','bagaimana',
    'kenapa','siapa','apa','mana','antara','tentang','sekarang',
    'dulu','sama','baru','lama','coba','orang','kamu','yah',
    'sini','disini','bakalan','seperti','apalagi','kemarin','pas',
    'kemana','wei','mbok','kek','ceu','jadi','tidak','ada','bukan',
    'yaa','yaah','yah','yahh','deh','dehh','dong','loh','lho',
    'sih','kok','kak','ka','kaa','kk','eh','ehh','ih','ihh',
    'hmm','wkwk','wkwkw','hehe','haha','huhu','weh','behh','duh',
    'aja','ajaa','aj','nih','ni','tuh','tu','lah','mah','dah',
    'gini','gitu','pun','ok','oke','sip','kok','sih','si',
    'kakak','pak','mas','min','bang','mbak','mbk','kalian',
    'the','and','is','it','in','of','to','you','your','so',
    'this','that','for','next','first','ever',
    'nadiem','nadiem makarim','chromebook','kasus','makarim',
]

kata_positif = [
    "bagus","baik","mantap","keren","suka","puas","senang","hebat","murah",
    "cepat","aman","sesuai","rekomen","oke","ramah","rapi","bersih",
    "asli","kuat","nyaman","menarik","percaya","cocok","tepat",
    "responsif","bantu","awet","lengkap","jujur","transparan","hasil",
    "kompeten","inovatif","edukatif","modern","maju","positif",
    "apresiasi","dukung","peduli","konsisten",
    "bebas","adil","benar","innocent","bela","ampun","lolos",
]
kata_negatif = [
    "buruk","jelek","rusak","hancur","kecewa","palsu","tipu","lambat","mahal","cacat",
    "kotor","lusuh","menyesal","bohong","lelet","telat","error","salah","kurang",
    "korupsi","koruptor","curang","gagal","parah","aneh","kacau","bodoh","jahat","kriminal",
    "rugi","malu","manipulatif","rakus","busuk","masalah","penjara","bukti","kriminalisasi",
    "paksa","janggal","rekayasa","politisasi","zalim",
]
kata_positif_bigram = [
    "bagus banget","sangat bagus","sangat puas","puas banget","cepat banget","harga murah",
    "kualitas bagus","mantap banget","oke banget","sangat baik","sangat bantu",
    "layak apresiasi","hukum adil",
]
kata_negatif_bigram = [
    "tidak bagus","tidak puas","tidak sesuai","tidak layak","tidak jelas","tidak nyaman",
    "kurang bagus","kurang puas","sangat kecewa","kecewa banget","tidak kompeten",
    "sangat parah","rugi negara","hukum tidak adil","kasus paksa",
]

@st.cache_resource
def load_model():
    with open('model_svm.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

@st.cache_data
def load_data():
    return pd.read_csv('nadiem_opini.csv')

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource
def load_stopwords_set():
    factory = StopWordRemoverFactory()
    sastrawi_sw = factory.get_stop_words()
    return set(sastrawi_sw + custom_stopwords)

def remove_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def preprocess(text):
    stemmer = load_stemmer()
    stopwords_set = load_stopwords_set()
    s = str(text).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r'@\w+|#\w+', ' ', s)
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = remove_repeated_chars(s)
    s = re.sub(r'\s+', ' ', s).strip()
    words = s.split()
    words = [slang_dict.get(w, w) for w in words]
    words = [w for w in words if w not in stopwords_set]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

# SIDEBAR 
st.sidebar.markdown("# 🔍 Search")
menu = st.sidebar.radio("Pilih Halaman", [
    "🏠 Beranda", "📊 Analisis Data", "☁️ WordCloud", "📈 N-Gram", "🤖 Prediksi Sentimen"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**👥 Kelompok 16**")
st.sidebar.markdown("Devi Roslidyanti")
st.sidebar.markdown("Sekar Ayu Nida Nur Afifah")

try:
    model, vectorizer = load_model()
    model_loaded = True
except:
    model_loaded = False

try:
    df_raw = load_data()
    data_loaded = True
except:
    data_loaded = False

colors = {'Negatif': '#FF4444', 'Netral': '#4488FF', 'Positif': '#44CC88'}

# BERANDA 
if menu == "🏠 Beranda":
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem 0;'>
        <h1 style='font-size:2.5rem; color:#FF6B00; border:none; padding:0;'>🔍 Analisis Sentimen Publik</h1>
        <p style='color:#aaaaaa; font-size:1.1rem;'>Kasus Dugaan Korupsi Chromebook Nadiem Makarim</p>
        <p style='color:#FF8C00; font-weight:700;'>Metode Support Vector Machine (SVM) • Media Sosial X</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if data_loaded and 'Sentimen_Lexicon' in df_raw.columns:
        dist = df_raw['Sentimen_Lexicon'].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total Data", f"{len(df_raw):,}")
        col2.metric("🔴 Negatif", dist.get('Negatif', 0))
        col3.metric("🔵 Netral", dist.get('Netral', 0))
        col4.metric("🟢 Positif", dist.get('Positif', 0))

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### 📌 Tentang Penelitian
        Penelitian ini menganalisis sentimen publik terhadap kasus dugaan korupsi 
        Chromebook Nadiem Makarim dari data media sosial X (Twitter) menggunakan 
        metode **Support Vector Machine (SVM)** dengan pelabelan **Lexicon-Based**.
        """)
    with col_b:
        st.markdown("""
        ### 🚀 Fitur Dashboard
        - 📊 Visualisasi distribusi sentimen
        - ☁️ WordCloud per kategori sentimen
        - 📈 Analisis N-Gram interaktif
        - 🤖 Prediksi sentimen real-time
        """)

# ANALISIS DATA 
elif menu == "📊 Analisis Data":
    st.title("📊 Analisis Data")
    if not data_loaded:
        st.error("File nadiem_opini.csv tidak ditemukan!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribusi Sentimen")
            if 'Sentimen_Lexicon' in df_raw.columns:
                dist = df_raw['Sentimen_Lexicon'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1a1a')
                ax.set_facecolor('#1a1a1a')
                bars = ax.bar(dist.index, dist.values,
                              color=[colors.get(k, 'gray') for k in dist.index],
                              edgecolor='none', width=0.5)
                for bar, val in zip(bars, dist.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                            str(val), ha='center', fontsize=11, fontweight='bold', color='white')
                ax.set_xlabel('Sentimen', color='#FF8C00', fontweight='bold')
                ax.set_ylabel('Jumlah', color='#FF8C00', fontweight='bold')
                ax.tick_params(colors='white')
                ax.spines[:].set_color('#333333')
                fig.tight_layout()
                st.pyplot(fig); plt.close()

        with col2:
            st.subheader("Persentase Sentimen")
            if 'Sentimen_Lexicon' in df_raw.columns:
                dist = df_raw['Sentimen_Lexicon'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1a1a')
                ax.set_facecolor('#1a1a1a')
                wedges, texts, autotexts = ax.pie(
                    dist.values, labels=dist.index, autopct='%1.1f%%',
                    colors=[colors.get(k, 'gray') for k in dist.index],
                    startangle=90, textprops={'color': 'white'})
                for at in autotexts:
                    at.set_fontweight('bold')
                fig.tight_layout()
                st.pyplot(fig); plt.close()

        st.markdown("---")
        st.subheader("🏆 Performa Model SVM (Kernel RBF)")
        col3, col4, col5, col6 = st.columns(4)
        col3.metric("Accuracy", "74.48%")
        col4.metric("Precision", "74.36%")
        col5.metric("Recall", "74.48%")
        col6.metric("F1-Score", "74.15%")
        st.info("**Kernel Terbaik: RBF** | Best Params: C=100, gamma=0.1 | CV Score: 0.6582 | Rata-rata CV Accuracy: 70.16%")

#  WORDCLOUD 
elif menu == "☁️ WordCloud":
    st.title("☁️ WordCloud")
    if not data_loaded:
        st.error("File nadiem_opini.csv tidak ditemukan!")
    elif 'filtered_text_stem' not in df_raw.columns:
        st.error("Kolom filtered_text_stem tidak ditemukan!")
    else:
        pilihan = st.selectbox("Pilih Kategori", ["Semua", "Positif", "Negatif", "Netral"])
        if pilihan == "Semua":
            subset = df_raw['filtered_text_stem'].dropna()
            cmap = 'YlOrRd'
            title = "WordCloud Semua Tweet"
        else:
            subset = df_raw[df_raw['Sentimen_Lexicon'] == pilihan]['filtered_text_stem'].dropna()
            cmap = {'Positif': 'Greens', 'Negatif': 'Reds', 'Netral': 'Blues'}[pilihan]
            title = f"WordCloud Sentimen {pilihan}"

        text_all = ' '.join(subset)
        if text_all.strip():
            wc = WordCloud(width=900, height=450, background_color='#1a1a1a',
                           colormap=cmap, max_words=150).generate(text_all)
            fig, ax = plt.subplots(figsize=(12, 5), facecolor='#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold', color='#FF8C00')
            fig.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.warning("Tidak ada data.")

# N-GRAM 
elif menu == "📈 N-Gram":
    st.title("📈 Analisis N-Gram")
    if not data_loaded:
        st.error("File nadiem_opini.csv tidak ditemukan!")
    elif 'filtered_text_stem' not in df_raw.columns:
        st.error("Kolom filtered_text_stem tidak ditemukan!")
    else:
        all_text = ' '.join(df_raw['filtered_text_stem'].dropna())
        all_words = all_text.split()
        top_n = st.slider("Tampilkan Top N", 10, 30, 20)

        def plot_ngram(data, title, color):
            words_ = [w if isinstance(w, str) else ' '.join(w) for w, _ in data]
            freqs_ = [c for _, c in data]
            fig, ax = plt.subplots(figsize=(9, 6), facecolor='#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            bars = ax.barh(words_[::-1], freqs_[::-1], color=color, edgecolor='none', height=0.6)
            ax.set_title(title, color='#FF8C00', fontweight='bold', fontsize=13)
            ax.set_xlabel('Frekuensi', color='#FF8C00')
            ax.tick_params(colors='white', labelsize=9)
            ax.spines[:].set_color('#333333')
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        tab1, tab2, tab3 = st.tabs(["Unigram", "Bigram", "Trigram"])
        with tab1:
            plot_ngram(Counter(all_words).most_common(top_n), 'Distribusi Unigram', '#44CC88')
        with tab2:
            plot_ngram(Counter(ngrams(all_words, 2)).most_common(top_n), 'Distribusi Bigram', '#FF6B00')
        with tab3:
            plot_ngram(Counter(ngrams(all_words, 3)).most_common(top_n), 'Distribusi Trigram', '#FF4444')

# PREDIKSI
elif menu == "🤖 Prediksi Sentimen":
    st.title("🤖 Prediksi Sentimen")
    st.markdown("Masukkan teks tweet untuk diprediksi sentimennya menggunakan model SVM.")

    if not model_loaded:
        st.error("Model tidak ditemukan!")
    else:
        teks_input = st.text_area("✏️ Masukkan Teks Tweet", height=150,
                                   placeholder="Contoh: Korupsi chromebook ini sangat merugikan negara...")
        if st.button("🔍 Prediksi Sekarang", type="primary"):
            if teks_input.strip():
                with st.spinner("Memproses..."):
                    teks_bersih = preprocess(teks_input)
                    X_input = vectorizer.transform([teks_bersih])
                    prediksi = model.predict(X_input)[0]
                    proba = model.predict_proba(X_input)[0]
                    classes = model.classes_

                st.markdown("---")
                st.subheader("Hasil Prediksi")
                if prediksi == 'Positif':
                    st.success(f"🟢 **Sentimen: {prediksi}**")
                elif prediksi == 'Negatif':
                    st.error(f"🔴 **Sentimen: {prediksi}**")
                else:
                    st.info(f"🔵 **Sentimen: {prediksi}**")

                st.markdown("**Probabilitas per kelas:**")
                for cls, prob in zip(classes, proba):
                    st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

                with st.expander("🔎 Detail Preprocessing"):
                    st.write(f"**Teks asli:** {teks_input}")
                    st.write(f"**Setelah preprocessing:** {teks_bersih}")
            else:
                st.warning("Masukkan teks terlebih dahulu!")
