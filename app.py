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

#  PAGE CONFIG
st.set_page_config(
    page_title="Analisis Sentimen SVM",
    page_icon="🔍",
    layout="wide"
)

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

# CUSTOM STOPWORDS 
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

#  LEXICON 
kata_positif = [
    "bagus","baik","mantap","keren","suka","puas","senang","hebat","murah",
    "cepat","aman","sesuai","rekomen","oke","ramah","rapi","bersih",
    "asli","kuat","nyaman","menarik","baik","percaya","cocok","tepat",
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

#  LOAD MODEL & DATA
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

#  PREPROCESSING 
def remove_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def preprocess(text):
    stemmer = load_stemmer()
    stopwords_set = load_stopwords_set()

    # cleaning
    s = str(text)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r'@\w+|#\w+', ' ', s)
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = remove_repeated_chars(s)
    s = re.sub(r'\s+', ' ', s).strip()

    # normalisasi slang
    words = s.split()
    words = [slang_dict.get(w, w) for w in words]

    # stopword removal
    words = [w for w in words if w not in stopwords_set]

    # stemming
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)

def hitung_sentimen_lexicon(text):
    if not text:
        return 0, "Netral"
    tokens = text.split()
    skor = 0
    for i in range(len(tokens)):
        kata = tokens[i]
        if kata in kata_positif:
            skor += 1
        elif kata in kata_negatif:
            skor -= 1
        if i < len(tokens) - 1:
            bigram_kata = kata + " " + tokens[i+1]
            if bigram_kata in kata_positif_bigram:
                skor += 2
            elif bigram_kata in kata_negatif_bigram:
                skor -= 2
    if skor > 0:
        label = "Positif"
    elif skor < 0:
        label = "Negatif"
    else:
        label = "Netral"
    return skor, label

#  SIDEBAR 
st.sidebar.title("🔍 Navigasi")
menu = st.sidebar.radio("Pilih Halaman", [
    "🏠 Beranda",
    "📊 Analisis Data",
    "☁️ WordCloud",
    "📈 N-Gram",
    "🤖 Prediksi Sentimen"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Author**")
st.sidebar.markdown("Sekar Ayu Nida Nur Afifah")
st.sidebar.markdown("Devi Roslidyanti")

#  LOAD 
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

colors = {'Negatif': 'tomato', 'Netral': 'steelblue', 'Positif': 'mediumseagreen'}

#  BERANDA 
if menu == "🏠 Beranda":
    st.title("🔍 Analisis Sentimen Publik")
    st.subheader("Kasus Dugaan Korupsi Chromebook Nadiem Makarim")
    st.markdown("Menggunakan Metode **Support Vector Machine (SVM)** pada Media Sosial X")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    if data_loaded and 'Sentimen_Lexicon' in df_raw.columns:
        dist = df_raw['Sentimen_Lexicon'].value_counts()
        col1.metric("Total Data", f"{len(df_raw):,}")
        col2.metric("🔴 Negatif", dist.get('Negatif', 0))
        col3.metric("🔵 Netral", dist.get('Netral', 0))
        col4.metric("🟢 Positif", dist.get('Positif', 0))

    st.markdown("---")
    st.markdown("""
    ### Tentang Dashboard
    Dashboard ini menampilkan hasil analisis sentimen publik terhadap kasus dugaan 
    korupsi Chromebook Nadiem Makarim yang dibahas di media sosial X (Twitter).
    
    **Fitur:**
    - 📊 Visualisasi distribusi sentimen
    - ☁️ WordCloud per kategori sentimen  
    - 📈 Analisis N-Gram (Unigram, Bigram, Trigram)
    - 🤖 Prediksi sentimen teks baru secara real-time
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
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(dist.index, dist.values,
                              color=[colors.get(k, 'gray') for k in dist.index])
                for bar, val in zip(bars, dist.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                            str(val), ha='center', fontsize=11, fontweight='bold')
                ax.set_xlabel('Sentimen')
                ax.set_ylabel('Jumlah')
                ax.set_title('Distribusi Label Sentimen')
                st.pyplot(fig)
                plt.close()

        with col2:
            st.subheader("Persentase Sentimen")
            if 'Sentimen_Lexicon' in df_raw.columns:
                dist = df_raw['Sentimen_Lexicon'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(dist.values, labels=dist.index, autopct='%1.1f%%',
                       colors=[colors.get(k, 'gray') for k in dist.index], startangle=90)
                ax.set_title('Persentase Label Sentimen')
                st.pyplot(fig)
                plt.close()

        st.markdown("---")
        st.subheader("Performa Model SVM (Kernel RBF)")
        col3, col4, col5, col6 = st.columns(4)
        col3.metric("Accuracy", "74.48%")
        col4.metric("Precision", "74.36%")
        col5.metric("Recall", "74.48%")
        col6.metric("F1-Score", "74.15%")
        st.markdown("**Kernel Terbaik: RBF** | Best Params: C=100, gamma=0.1 | CV Score: 0.6582")
        st.markdown("**Rata-rata Cross Validation:** Accuracy 70.16% | F1-Score 69.75%")

# WORDCLOUD
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
            cmap = 'viridis'
            title = "WordCloud Semua Tweet"
        else:
            subset = df_raw[df_raw['Sentimen_Lexicon'] == pilihan]['filtered_text_stem'].dropna()
            cmap = {'Positif': 'Greens', 'Negatif': 'Reds', 'Netral': 'Blues'}[pilihan]
            title = f"WordCloud Sentimen {pilihan}"

        text_all = ' '.join(subset)
        if text_all.strip():
            wc = WordCloud(width=900, height=450, background_color='white',
                           colormap=cmap, max_words=150).generate(text_all)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Tidak ada data untuk ditampilkan.")

#  N-GRAM 
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

        tab1, tab2, tab3 = st.tabs(["Unigram", "Bigram", "Trigram"])
        with tab1:
            data = Counter(all_words).most_common(top_n)
            words_, freqs_ = zip(*data)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(list(words_)[::-1], list(freqs_)[::-1], color='mediumseagreen')
            ax.set_title('Distribusi Unigram')
            ax.set_xlabel('Frekuensi')
            st.pyplot(fig); plt.close()

        with tab2:
            data = Counter(ngrams(all_words, 2)).most_common(top_n)
            labels = [' '.join(w) for w, _ in data]
            freqs_ = [c for _, c in data]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(labels[::-1], freqs_[::-1], color='steelblue')
            ax.set_title('Distribusi Bigram')
            ax.set_xlabel('Frekuensi')
            st.pyplot(fig); plt.close()

        with tab3:
            data = Counter(ngrams(all_words, 3)).most_common(top_n)
            labels = [' '.join(w) for w, _ in data]
            freqs_ = [c for _, c in data]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(labels[::-1], freqs_[::-1], color='tomato')
            ax.set_title('Distribusi Trigram')
            ax.set_xlabel('Frekuensi')
            st.pyplot(fig); plt.close()

# PREDIKSI 
elif menu == "🤖 Prediksi Sentimen":
    st.title("🤖 Prediksi Sentimen")
    st.markdown("Masukkan teks tweet untuk diprediksi sentimennya menggunakan model SVM.")

    if not model_loaded:
        st.error("Model tidak ditemukan! Pastikan file model_svm.pkl dan tfidf_vectorizer.pkl ada.")
    else:
        teks_input = st.text_area("✏️ Masukkan Teks Tweet", height=150,
                                   placeholder="Contoh: Korupsi chromebook ini sangat merugikan negara...")
        if st.button("🔍 Prediksi", type="primary"):
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
                    st.success(f"🟢 Sentimen: **{prediksi}**")
                elif prediksi == 'Negatif':
                    st.error(f"🔴 Sentimen: **{prediksi}**")
                else:
                    st.info(f"🔵 Sentimen: **{prediksi}**")

                st.markdown("**Probabilitas per kelas:**")
                for cls, prob in zip(classes, proba):
                    st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

                with st.expander("Lihat hasil preprocessing"):
                    st.write(f"**Teks asli:** {teks_input}")
                    st.write(f"**Teks setelah preprocessing:** {teks_bersih}")
            else:
                st.warning("Masukkan teks terlebih dahulu!")
