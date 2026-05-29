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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(
    page_title="Analisis Sentimen Kasus Dugaan Korupsi Chromebook Nadiem Makarim dengan SVM",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #F8FAFC; }
[data-testid="stSidebar"] { background-color: #1E293B; border-right: none; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stRadio label { color: #94A3B8 !important; font-size: 0.9rem !important; }
[data-testid="stSidebar"] hr { border-color: #334155 !important; }
h1 { color: #0F172A !important; font-weight: 700 !important; font-size: 1.8rem !important; border: none !important; padding-bottom: 0 !important; }
h2, h3, h4 { color: #1E293B !important; font-weight: 600 !important; }
p, li { color: #475569 !important; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.85rem !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }
.stButton > button { background-color: #2563EB !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 500 !important; padding: 0.5rem 1.5rem !important; }
.stButton > button:hover { background-color: #1D4ED8 !important; }
.stSelectbox > div > div { background-color: #FFFFFF !important; border: 1px solid #CBD5E1 !important; border-radius: 8px !important; color: #0F172A !important; }
.stSelectbox label { color: #1E293B !important; font-weight: 500 !important; }
div[data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #0F172A !important; }
.stTextArea > div > div > textarea { background-color: #FFFFFF !important; border: 1px solid #CBD5E1 !important; border-radius: 8px !important; color: #0F172A !important; }
.stTabs [data-baseweb="tab-list"] { background-color: #F1F5F9; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #64748B !important; font-weight: 500 !important; border-radius: 7px !important; font-size: 0.9rem !important; }
.stTabs [aria-selected="true"] { background-color: #FFFFFF !important; color: #0F172A !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important; }
.stSuccess { background-color: #F0FDF4 !important; border-left: 3px solid #22C55E !important; border-radius: 8px !important; }
.stError { background-color: #FFF1F2 !important; border-left: 3px solid #EF4444 !important; border-radius: 8px !important; }
.stInfo { background-color: #EFF6FF !important; border-left: 3px solid #2563EB !important; border-radius: 8px !important; }
.interpretasi { background: #F0F9FF; border-left: 3px solid #2563EB; border-radius: 6px; padding: 0.8rem 1rem; margin-top: 0.5rem; color: #1E3A5F !important; font-size: 0.9rem; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── SLANG DICT ───────────────────────────────────────────────────────────────
slang_dict = {
    "sy":"saya","sya":"saya","aq":"saya","ak":"saya","gw":"saya","gue":"saya","gua":"saya","w":"saya",
    "km":"kamu","lo":"kamu","loe":"kamu","lu":"kamu",
    "gak":"tidak","ga":"tidak","gk":"tidak","ngga":"tidak","nggak":"tidak","ngak":"tidak",
    "kaga":"tidak","tdk":"tidak","tdak":"tidak","dak":"tidak","ndak":"tidak","ndk":"tidak",
    "gaada":"tidak ada","gada":"tidak ada","gabisa":"tidak bisa","gaboleh":"tidak boleh",
    "gamau":"tidak mau","gasuka":"tidak suka","gapeduli":"tidak peduli","gajelas":"tidak jelas",
    "gabener":"tidak benar","gasetuju":"tidak setuju","gaksetuju":"tidak setuju",
    "gatau":"tidak tahu","gtau":"tidak tahu","gaktau":"tidak tahu","gausah":"tidak usah","jgn":"jangan",
    "udah":"sudah","udh":"sudah","uda":"sudah","sdh":"sudah","blm":"belum","blom":"belum","belom":"belum",
    "yg":"yang","yng":"yang","dgn":"dengan","dg":"dengan","krn":"karena","karna":"karena",
    "utk":"untuk","tp":"tapi","tpi":"tapi","jg":"juga","jga":"juga","sm":"sama",
    "dr":"dari","dri":"dari","drpd":"daripada","dlm":"dalam","pd":"pada","ttg":"tentang","bhw":"bahwa","sbg":"sebagai",
    "bgt":"banget","sgt":"sangat","lg":"lagi","dl":"dulu","skrg":"sekarang","trs":"terus",
    "ttp":"tetap","lbh":"lebih","bnyk":"banyak","byk":"banyak","bs":"bisa","dpt":"dapat",
    "hrs":"harus","msh":"masih","tau":"tahu","liat":"lihat",
    "mikir":"berpikir","males":"malas","maksa":"memaksa","denger":"dengar",
    "ngajak":"mengajak","ngerti":"mengerti","ngeluh":"mengeluh","ngomong":"berbicara","ngerasa":"merasa",
    "ngebela":"membela","ngebahas":"membahas","nyuruh":"menyuruh","nyari":"mencari",
    "korup":"korupsi","bag":"bagian",
    "nih":"ini","ni":"ini","tuh":"itu","tu":"itu","gitu":"begitu","gimana":"bagaimana",
    "klo":"kalau","kalo":"kalau","ky":"kayak","kyk":"kayak",
    "bener":"benar","emg":"memang","emang":"memang","aja":"saja","doang":"saja",
    "org":"orang","makin":"semakin","blio":"beliau",
    "tipikor":"tindak pidana korupsi","uu":"undang undang",
    "mksh":"terima kasih","makasih":"terima kasih","tq":"terima kasih","thx":"terima kasih",
}

custom_stopwords = [
    'yang','dan','di','ke','dari','ini','itu','dengan','untuk','adalah','atau','juga',
    'dalam','sudah','saya','aku','ges','ku','kami','kita','mereka','dia','ia','anda',
    'akan','dapat','pada','oleh','karena','namun','tetapi','tapi','jika','kalau',
    'maka','agar','mau','bagi','nya','pun','lebih','sangat','sekali','hanya','saja',
    'ya','lagi','masih','harus','boleh','memang','terus','lalu','setelah','sebelum',
    'saat','ketika','pernah','cara','hal','banyak','beberapa','semua','setiap',
    'tersebut','seperti','bahwa','tolong','bagaimana','kenapa','siapa','apa','mana',
    'antara','tentang','sekarang','dulu','sama','baru','lama','coba','orang','kamu',
    'sini','disini','bakalan','apalagi','kemarin','pas','jadi','tidak','ada','bukan',
    'yaa','yah','deh','dong','loh','sih','kok','kak','ka','eh','hmm',
    'aja','nih','ni','tuh','tu','lah','ok','oke','sip','si',
    'kakak','pak','mas','min','bang','mbak','kalian',
    'the','and','is','it','in','of','to','you','your','so','this','that','for',
    'nadiem','chromebook','kasus','makarim',
]

kata_positif = [
    "bagus","baik","mantap","keren","suka","puas","senang","hebat","murah","cepat","aman",
    "sesuai","rekomen","oke","ramah","rapi","bersih","asli","kuat","nyaman","menarik",
    "percaya","cocok","tepat","responsif","bantu","awet","lengkap","jujur","transparan",
    "kompeten","inovatif","edukatif","modern","maju","positif","apresiasi","dukung",
    "peduli","konsisten","bebas","adil","benar","innocent","bela","ampun","lolos",
]
kata_negatif = [
    "buruk","jelek","rusak","hancur","kecewa","palsu","tipu","lambat","mahal","cacat",
    "kotor","lusuh","menyesal","bohong","lelet","telat","error","salah","kurang",
    "korupsi","koruptor","curang","gagal","parah","aneh","kacau","bodoh","jahat","kriminal",
    "rugi","malu","manipulatif","rakus","busuk","masalah","penjara","bukti","kriminalisasi",
    "paksa","janggal","rekayasa","politisasi","zalim",
]
kata_positif_bigram = [
    "bagus banget","sangat bagus","sangat puas","harga murah","kualitas bagus",
    "sangat baik","sangat bantu","layak apresiasi","hukum adil",
]
kata_negatif_bigram = [
    "tidak bagus","tidak puas","tidak sesuai","tidak layak","tidak jelas",
    "kurang bagus","sangat kecewa","tidak kompeten","sangat parah",
    "rugi negara","hukum tidak adil","kasus paksa",
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
    return StemmerFactory().create_stemmer()

@st.cache_resource
def load_stopwords_set():
    sastrawi_sw = StopWordRemoverFactory().get_stop_words()
    return set(sastrawi_sw + custom_stopwords)

def preprocess(text):
    stemmer = load_stemmer()
    sw = load_stopwords_set()
    s = str(text).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r'@\w+|#\w+|http\S+|www\.\S+', ' ', s)
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'(.)\1{2,}', r'\1\1', s)
    s = re.sub(r'\s+', ' ', s).strip()
    words = [slang_dict.get(w, w) for w in s.split()]
    words = [stemmer.stem(w) for w in words if w not in sw]
    return ' '.join(words)

def interpretasi(teks):
    st.markdown(f'<div class="interpretasi">💡 <b>Interpretasi:</b> {teks}</div>', unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("### MENU")
menu = st.sidebar.radio("", [
    "Beranda", "Analisis Data", "Evaluasi Model", "WordCloud", "N-Gram", "Prediksi Sentimen"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Kelompok 16**")
st.sidebar.caption("Devi Roslidyanti · 23031030024")
st.sidebar.caption("Sekar Ayu Nida Nur Afifah · 23031030028")

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

PLT_BG = '#FFFFFF'
colors = {'Negatif': '#EF4444', 'Netral': '#3B82F6', 'Positif': '#22C55E'}

def styled_plot(figsize=(6,4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=PLT_BG)
    ax.set_facecolor(PLT_BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E2E8F0')
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.tick_params(colors='#475569', labelsize=9)
    ax.yaxis.label.set_color('#475569')
    ax.xaxis.label.set_color('#475569')
    return fig, ax

# ─── BERANDA ──────────────────────────────────────────────────────────────────
if menu == "Beranda":
    st.markdown("## Analisis Sentimen Publik")
    st.markdown("**Kasus Dugaan Korupsi Chromebook Nadiem Makarim pada Media Sosial X**")
    st.markdown("Menggunakan metode Support Vector Machine (SVM) dengan pelabelan Lexicon-Based")
    st.divider()

    if data_loaded and 'Sentimen_Lexicon' in df_raw.columns:
        dist = df_raw['Sentimen_Lexicon'].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data", f"{len(df_raw):,}")
        c2.metric("Negatif", dist.get('Negatif', 0))
        c3.metric("Netral", dist.get('Netral', 0))
        c4.metric("Positif", dist.get('Positif', 0))

    st.divider()
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown("#### Tentang Penelitian")
        st.markdown("""
        Penelitian ini menganalisis opini publik di media sosial X terkait kasus 
        dugaan korupsi pengadaan Chromebook oleh Nadiem Makarim. Data diproses 
        melalui tahapan text preprocessing, kemudian diklasifikasikan menggunakan 
        algoritma SVM dengan representasi fitur TF-IDF.
        """)
    with c_b:
        st.markdown("#### Fitur Dashboard")
        st.markdown("""
        - Distribusi dan visualisasi data sentimen  
        - Evaluasi model (Confusion Matrix & Kurva ROC)
        - WordCloud per kategori sentimen  
        - Analisis frekuensi N-Gram  
        - Prediksi sentimen teks baru secara real-time  
        """)

# ─── ANALISIS DATA ─────────────────────────────────────────────────────────────
elif menu == "Analisis Data":
    st.markdown("## Analisis Data")
    if not data_loaded:
        st.error("File nadiem_opini.csv tidak ditemukan.")
    else:
        # Distribusi sentimen
        st.markdown("#### Distribusi Label Sentimen")
        st.markdown("Berikut merupakan distribusi label sentimen pada dataset berdasarkan hasil pelabelan otomatis menggunakan metode Lexicon-Based.")
        c1, c2 = st.columns(2)
        with c1:
            if 'Sentimen_Lexicon' in df_raw.columns:
                dist = df_raw['Sentimen_Lexicon'].value_counts()
                fig, ax = styled_plot((6, 4))
                bars = ax.bar(dist.index, dist.values,
                              color=[colors.get(k, '#94A3B8') for k in dist.index],
                              width=0.45, edgecolor='none')
                for bar, val in zip(bars, dist.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                            str(val), ha='center', fontsize=10, fontweight='600', color='#1E293B')
                ax.set_title('Distribusi Label Sentimen', fontsize=12, fontweight='600', color='#1E293B', pad=10)
                ax.set_ylabel('Jumlah Tweet')
                fig.tight_layout()
                st.pyplot(fig); plt.close()

        with c2:
            if 'Sentimen_Lexicon' in df_raw.columns:
                dist = df_raw['Sentimen_Lexicon'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4), facecolor=PLT_BG)
                ax.set_facecolor(PLT_BG)
                wedges, texts, autotexts = ax.pie(
                    dist.values, labels=dist.index, autopct='%1.1f%%',
                    colors=[colors.get(k, '#94A3B8') for k in dist.index],
                    startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
                for t in texts: t.set_color('#475569'); t.set_fontsize(10)
                for at in autotexts: at.set_fontweight('600'); at.set_fontsize(9)
                ax.set_title('Proporsi Label Sentimen', fontsize=12, fontweight='600', color='#1E293B', pad=10)
                fig.tight_layout()
                st.pyplot(fig); plt.close()

        interpretasi("Dari total 1.193 data, mayoritas tweet bersifat Negatif (529 data / 44,3%), diikuti Netral (430 data / 36,0%), dan Positif (234 data / 19,6%). Dominannya sentimen negatif mengindikasikan bahwa publik di media sosial X cenderung memberikan respons yang tidak mendukung terhadap kasus dugaan korupsi Chromebook Nadiem Makarim.")

        st.divider()

        # Distribusi jumlah karakter & kata
        if 'jumlah_karakter' in df_raw.columns and 'jumlah_kata' in df_raw.columns:
            st.markdown("#### Karakteristik Panjang Tweet")
            st.markdown("Berikut merupakan distribusi panjang tweet berdasarkan jumlah karakter dan jumlah kata per tweet.")
            c3, c4 = st.columns(2)
            with c3:
                fig, ax = styled_plot((6, 4))
                ax.hist(df_raw['jumlah_karakter'].dropna(), bins=30, color='#3B82F6', edgecolor='white', linewidth=0.5)
                ax.set_title('Distribusi Jumlah Karakter per Tweet', fontsize=11, fontweight='600', color='#1E293B', pad=10)
                ax.set_xlabel('Jumlah Karakter')
                ax.set_ylabel('Frekuensi')
                fig.tight_layout()
                st.pyplot(fig); plt.close()

            with c4:
                fig, ax = styled_plot((6, 4))
                ax.hist(df_raw['jumlah_kata'].dropna(), bins=30, color='#8B5CF6', edgecolor='white', linewidth=0.5)
                ax.set_title('Distribusi Jumlah Kata per Tweet', fontsize=11, fontweight='600', color='#1E293B', pad=10)
                ax.set_xlabel('Jumlah Kata')
                ax.set_ylabel('Frekuensi')
                fig.tight_layout()
                st.pyplot(fig); plt.close()

            interpretasi("Distribusi jumlah karakter dan kata per tweet menunjukkan pola right-skewed (condong ke kanan), artinya sebagian besar tweet memiliki panjang yang relatif pendek yaitu antara 50–200 karakter atau 10–50 kata per tweet. Hanya sedikit tweet yang memiliki panjang ekstrem, kemungkinan merupakan thread atau quote tweet yang panjang.")

        st.divider()
        st.markdown("#### Performa Model SVM — Kernel RBF")
        st.markdown("Berikut merupakan ringkasan hasil evaluasi model SVM dengan kernel terbaik yang dipilih berdasarkan GridSearchCV.")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Accuracy", "74.48%")
        c6.metric("Precision", "74.36%")
        c7.metric("Recall", "74.48%")
        c8.metric("F1-Score", "74.15%")
        st.info("Kernel terbaik: **RBF** (C=100, gamma=0.1) · CV Score: 0.6582 · Rata-rata CV Accuracy: 70.16%")
        interpretasi("Model SVM dengan kernel RBF berhasil memprediksi 178 dari 239 data uji dengan benar (akurasi 74,48%). Kernel RBF dipilih sebagai terbaik berdasarkan best score cross-validation tertinggi (0,6582) dibandingkan kernel Linear dan Polynomial yang sama-sama menghasilkan score 0,6561.")

# ─── EVALUASI MODEL ────────────────────────────────────────────────────────────
elif menu == "Evaluasi Model":
    st.markdown("## Evaluasi Model")
    st.markdown("Halaman ini menampilkan hasil evaluasi mendalam model SVM kernel RBF melalui Confusion Matrix dan Kurva ROC.")

    if not model_loaded or not data_loaded:
        st.error("Model atau data tidak ditemukan.")
    else:
        t1, t2 = st.tabs(["Confusion Matrix", "Kurva ROC"])

        with t1:
            st.markdown("#### Confusion Matrix — Kernel RBF")
            st.markdown("Confusion matrix menampilkan perbandingan antara label prediksi model dengan label sebenarnya.")
            if 'filtered_text_stem' in df_raw.columns and 'Sentimen_Lexicon' in df_raw.columns:
                with st.spinner("Memuat confusion matrix..."):
                    from sklearn.model_selection import train_test_split
                    df_eval = df_raw[['filtered_text_stem','Sentimen_Lexicon']].dropna()
                    _, X_test_raw, _, y_test = train_test_split(
                        df_eval['filtered_text_stem'], df_eval['Sentimen_Lexicon'],
                        test_size=0.2, random_state=42, stratify=df_eval['Sentimen_Lexicon']
                    )
                    X_test_tfidf = vectorizer.transform(X_test_raw)
                    y_pred = model.predict(X_test_tfidf)
                    cm = confusion_matrix(y_test, y_pred, labels=['Negatif','Netral','Positif'])

                fig, ax = plt.subplots(figsize=(6, 5), facecolor=PLT_BG)
                ax.set_facecolor(PLT_BG)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negatif','Netral','Positif'])
                disp.plot(ax=ax, colorbar=True, cmap='Blues')
                ax.set_title('Confusion Matrix Kernel RBF', fontsize=12, fontweight='600', color='#1E293B', pad=12)
                fig.tight_layout()
                st.pyplot(fig); plt.close()

                interpretasi("Kelas Negatif memiliki prediksi terbaik dengan 92 dari 106 data berhasil diklasifikasikan dengan benar (86,8%). Kelas Positif memiliki performa terendah — dari 47 data, hanya 26 yang diprediksi benar (55,3%), dan 19 di antaranya salah diklasifikasikan sebagai Netral. Hal ini kemungkinan karena tweet positif dan netral memiliki konteks yang mirip sehingga sulit dibedakan oleh model.")

        with t2:
            st.markdown("#### Kurva ROC — Kernel RBF")
            st.markdown("Kurva ROC menggambarkan kemampuan model dalam membedakan antar kelas sentimen pada berbagai threshold.")
            if 'filtered_text_stem' in df_raw.columns and 'Sentimen_Lexicon' in df_raw.columns:
                with st.spinner("Memuat kurva ROC..."):
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import label_binarize
                    from sklearn.metrics import roc_curve, auc

                    df_eval = df_raw[['filtered_text_stem','Sentimen_Lexicon']].dropna()
                    classes = ['Negatif', 'Netral', 'Positif']
                    _, X_test_raw, _, y_test = train_test_split(
                        df_eval['filtered_text_stem'], df_eval['Sentimen_Lexicon'],
                        test_size=0.2, random_state=42, stratify=df_eval['Sentimen_Lexicon']
                    )
                    X_test_tfidf = vectorizer.transform(X_test_raw)
                    y_score = model.predict_proba(X_test_tfidf)
                    y_test_bin = label_binarize(y_test, classes=classes)

                roc_colors = {'Negatif': '#EF4444', 'Netral': '#22C55E', 'Positif': '#3B82F6'}
                fig, ax = styled_plot((7, 5))
                for i, cls in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    auc_score = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=roc_colors[cls], lw=2,
                            label=f'{cls} (AUC = {auc_score:.3f})')
                ax.plot([0,1],[0,1], 'k--', lw=1, label='Random Classifier')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Kurva ROC Kernel RBF', fontsize=12, fontweight='600', color='#1E293B')
                ax.legend(loc='lower right', fontsize=9)
                fig.tight_layout()
                st.pyplot(fig); plt.close()

                interpretasi("Kurva ROC model SVM kernel RBF menunjukkan performa klasifikasi yang baik pada semua kelas. Kelas Negatif memiliki AUC tertinggi (0,941), diikuti Positif (0,922), dan Netral (0,820). Nilai AUC di atas 0,8 untuk semua kelas menunjukkan model memiliki kemampuan diskriminasi yang baik. Kelas Netral memiliki AUC terendah, konsisten dengan hasil evaluasi yang menunjukkan kelas ini paling sulit dibedakan dari kelas lainnya.")

# ─── WORDCLOUD ─────────────────────────────────────────────────────────────────
elif menu == "WordCloud":
    st.markdown("## WordCloud")
    st.markdown("WordCloud menampilkan kata-kata yang paling sering muncul dalam dataset, di mana ukuran kata mencerminkan frekuensi kemunculannya.")
    if not data_loaded:
        st.error("File nadiem_opini.csv tidak ditemukan.")
    elif 'filtered_text_stem' not in df_raw.columns:
        st.error("Kolom filtered_text_stem tidak ditemukan.")
    else:
        st.markdown("**Pilih kategori sentimen** untuk menampilkan WordCloud yang sesuai.")
        pilihan = st.selectbox("Kategori Sentimen", ["Semua", "Positif", "Negatif", "Netral"],
                               key="wc_select")
        cmap_map = {"Semua": "Blues", "Positif": "Greens", "Negatif": "Reds", "Netral": "PuBu"}
        subset = df_raw['filtered_text_stem'].dropna() if pilihan == "Semua" \
            else df_raw[df_raw['Sentimen_Lexicon'] == pilihan]['filtered_text_stem'].dropna()

        text_all = ' '.join(subset)
        if text_all.strip():
            wc = WordCloud(width=1000, height=450, background_color='#FFFFFF',
                           colormap=cmap_map[pilihan], max_words=150,
                           collocations=False).generate(text_all)
            fig, ax = plt.subplots(figsize=(12, 5), facecolor=PLT_BG)
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f"WordCloud — Sentimen {pilihan}", fontsize=13,
                        fontweight='600', color='#1E293B', pad=12)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

            interp_map = {
                "Semua": "WordCloud keseluruhan didominasi oleh kata jaksa, ibam, hukum, korupsi, sidang, dan tuntut. Hal ini mempertegas bahwa percakapan publik di media X sangat didominasi oleh konteks persidangan dan proses hukum, bukan aspek teknis pengadaan Chromebook itu sendiri.",
                "Positif": "Pada kelompok sentimen positif, kata yang menonjol adalah hukum, adil, benar, baik, saksi, dan sidang. Tweet positif cenderung mengekspresikan harapan terhadap proses hukum yang adil serta dukungan terhadap penegakan hukum.",
                "Negatif": "WordCloud sentimen negatif didominasi kata korupsi, bukti, hukum, ibam, jaksa, dan rugi negara. Tweet negatif mencerminkan kekecewaan dan kemarahan publik terhadap dugaan korupsi serta sorotan terhadap kerugian negara.",
                "Netral": "Pada kelompok netral, kata dominan adalah sidang, lihat, ibam, hukum, dan jokowi. Tweet netral cenderung berupa laporan atau pengamatan tanpa keberpihakan, seperti pemberitaan jalannya sidang dan pemantauan perkembangan kasus.",
            }
            interpretasi(interp_map[pilihan])

# ─── N-GRAM ────────────────────────────────────────────────────────────────────
elif menu == "N-Gram":
    st.markdown("## Analisis N-Gram")
    st.markdown("Analisis N-Gram dilakukan untuk mengetahui pola kata dan frasa yang paling sering muncul dalam dataset.")
    if not data_loaded:
        st.error("File nadiem_opini.csv tidak ditemukan.")
    elif 'filtered_text_stem' not in df_raw.columns:
        st.error("Kolom filtered_text_stem tidak ditemukan.")
    else:
        all_words = ' '.join(df_raw['filtered_text_stem'].dropna()).split()
        top_n = st.slider("Jumlah kata/frasa yang ditampilkan", 10, 30, 20)

        def plot_ng(data, title, color, interp_text):
            labels = [w if isinstance(w, str) else ' '.join(w) for w, _ in data]
            freqs = [c for _, c in data]
            fig, ax = styled_plot((9, max(4, top_n * 0.35)))
            ax.barh(labels[::-1], freqs[::-1], color=color, height=0.55, edgecolor='none')
            ax.set_title(title, fontsize=12, fontweight='600', color='#1E293B', pad=10)
            ax.set_xlabel('Frekuensi')
            fig.tight_layout()
            st.pyplot(fig); plt.close()
            interpretasi(interp_text)

        t1, t2, t3 = st.tabs(["Unigram", "Bigram", "Trigram"])
        with t1:
            plot_ng(
                Counter(all_words).most_common(top_n),
                'Distribusi Unigram',
                '#3B82F6',
                "Kata yang paling dominan muncul adalah 'korupsi' dengan frekuensi lebih dari 530 kemunculan, jauh melampaui kata lainnya. Kata-kata dominan lainnya meliputi 'hukum', 'ada', 'jaksa', 'sidang', dan 'ibam', yang mencerminkan bahwa diskusi publik sangat berfokus pada aspek hukum dan proses persidangan kasus ini."
            )
        with t2:
            plot_ng(
                Counter(ngrams(all_words, 2)).most_common(top_n),
                'Distribusi Bigram',
                '#22C55E',
                "Pasangan kata yang paling sering muncul adalah 'duga korupsi' (±118 kali), diikuti 'korupsi ada' dan 'rugi negara' (±110 kali). Bigram dominan lainnya seperti 'tindak pidana', 'rupiah triliun', dan 'tom lembong' menunjukkan bahwa publik banyak membicarakan aspek kerugian negara dan tokoh-tokoh yang terlibat."
            )
        with t3:
            plot_ng(
                Counter(ngrams(all_words, 3)).most_common(top_n),
                'Distribusi Trigram',
                '#F59E0B',
                "Trigram paling dominan adalah 'duga korupsi ada' (±82 kali) dan 'tindak pidana korupsi' (±75 kali), yang mempertegas bahwa narasi utama di media X berpusat pada dugaan tindak pidana korupsi. Trigram 'korupsi ada laptop' dan 'adil tindak pidana' menunjukkan diskusi spesifik terkait pengadaan laptop Chromebook dan tuntutan keadilan hukum."
            )

# ─── PREDIKSI ──────────────────────────────────────────────────────────────────
elif menu == "Prediksi Sentimen":
    st.markdown("## Prediksi Sentimen")
    st.markdown("Masukkan teks tweet untuk diprediksi sentimennya secara real-time menggunakan model SVM yang telah dilatih.")
    st.divider()

    if not model_loaded:
        st.error("Model tidak ditemukan.")
    else:
        teks_input = st.text_area("Teks Tweet", height=140,
                                   placeholder="Tulis atau tempel teks tweet di sini...")
        if st.button("Prediksi"):
            if teks_input.strip():
                with st.spinner("Memproses teks..."):
                    teks_bersih = preprocess(teks_input)
                    X_in = vectorizer.transform([teks_bersih])
                    prediksi = model.predict(X_in)[0]
                    proba = model.predict_proba(X_in)[0]
                    classes = model.classes_

                st.divider()
                if prediksi == 'Positif':
                    st.success(f"Hasil Prediksi: **{prediksi}**")
                elif prediksi == 'Negatif':
                    st.error(f"Hasil Prediksi: **{prediksi}**")
                else:
                    st.info(f"Hasil Prediksi: **{prediksi}**")

                st.markdown("**Probabilitas per kelas**")
                for cls, prob in zip(classes, proba):
                    st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

                with st.expander("Detail preprocessing"):
                    st.caption(f"Teks asli: {teks_input}")
                    st.caption(f"Setelah preprocessing: {teks_bersih}")
            else:
                st.warning("Masukkan teks terlebih dahulu.")
