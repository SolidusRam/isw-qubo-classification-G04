import sys
import os

# Assicuriamoci che python trovi la directory 'src' indipendentemente da dove lanciamo il comando
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

def scroll_to_bottom():
    js = '''
    <script>
        setTimeout(function() {
            var mainContainer = window.parent.document.querySelector('section.main') || window.parent.document.querySelector('.main');
            if (mainContainer) {
                mainContainer.scrollTo({top: mainContainer.scrollHeight, behavior: 'smooth'});
            }
        }, 300);
    </script>
    '''
    components.html(js, height=0)

from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train, predict

st.set_page_config(page_title="QUBO Binary Classification", page_icon="⚛️", layout="wide")

st.markdown("""
    <style>
    /* Centra le tab e ingrandisce il font */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}

def main():

    tab_home, tab_prep, tab_fs, tab_train, tab_pred = st.tabs([
        "🏠 Home", "1️⃣ Preprocessing", "2️⃣ Feature Selection", "3️⃣ Training", "4️⃣ Prediction"
    ])

    with tab_home:
        st.title("Progetto di Classificazione Binaria con QUBO")
        st.markdown("""
        Questa applicazione permette di eseguire l'intera pipeline di elaborazione, riduzione delle feature e addestramento
        utilizzando algoritmi di classificazione binaria potenziati con feature selection via QUBO.
        
        **Fasi della pipeline:**
        1. **Preprocessing**: Caricamento, pulizia e normalizzazione (Z-score) dei dati.
        2. **Feature Selection**: Ottimizzazione QUBO per la scelta delle migliori features tramite matrice di correlazione Spearman.
        3. **Training**: Addestramento del classificatore usando le feature selezionate.
        4. **Prediction**: Predizione su un dataset di test con risultati dettagliati.
        """)

    with tab_prep:
        st.header("Fase 1: Preprocessing del Dataset")
        st.write("Questa fase normalizza il dataset ed elimina le colonne quasi vuote.")
        
        data_source = st.radio("Metodo di caricamento Dataset:", ["Percorso Manuale", "Upload File"], horizontal=True)
        
        input_csv = ""
        if data_source == "Upload File":
            uploaded_file = st.file_uploader("Carica il dataset CSV", type=["csv"])
            if uploaded_file is not None:
                input_csv = os.path.join(OUTPUT_DIR, "temp_uploaded.csv")
                with open(input_csv, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("File caricato!")
        else:
            input_csv = st.text_input("Inserisci il percorso al file CSV", value="data/sample_test_dataset.csv")

        col_a, col_b = st.columns(2)
        with col_a:
            target_col = st.text_input("Nome colonna Target", "target", key="prep_target")
        with col_b:
            min_perc_valid = st.slider("Percentuale minima dati validi (es. 0.05 = 5%)", 0.01, 1.0, 0.05, step=0.01)
        
        with st.expander("⚙️ Impostazioni Avanzate: File di Output"):
            out_csv = st.text_input("Nome CSV di output", os.path.join(OUTPUT_DIR, "normalized.csv"))
            out_json = st.text_input("Nome JSON di statistiche", os.path.join(OUTPUT_DIR, "preprocessing_result.json"))

        if st.button("Avvia Preprocessing", type="primary", use_container_width=True):
            if not input_csv or not os.path.exists(input_csv):
                st.error("Il file di input non esiste o non è stato caricato!")
            elif not target_col:
                st.error("Inserisci il nome della colonna target!")
            else:
                with st.spinner("Elaborazione in corso..."):
                    try:
                        fit_normalize(
                            input_csv=input_csv,
                            target_column=target_col,
                            normalized_csv=out_csv,
                            outInitalRes_json=out_json,
                            minPercValid=min_perc_valid
                        )
                        st.success("Preprocessing completato con successo!")
                        scroll_to_bottom()
                        
                        stats = load_json(out_json)
                        if stats:
                            with st.container(border=True):
                                st.subheader("🎯 Risultati Preprocessing")
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Features Iniziali", stats.get("n_input_features", 0))
                                c2.metric("Features Mantenute", stats.get("n_kept_features", 0), delta=f"-{stats.get('n_input_features', 0) - stats.get('n_kept_features', 0)}", delta_color="inverse")
                                c3.metric("Campioni", stats.get("dataset_size", 0))
                                
                                # Grafico a torta delle feature
                                fig = px.pie(
                                    names=['Mantenute', 'Rimosse'],
                                    values=[stats.get("n_kept_features", 0), stats.get('n_input_features', 0) - stats.get('n_kept_features', 0)],
                                    title="Distribuzione Features",
                                    hole=0.4,
                                    color_discrete_sequence=['#00d4ff', '#1e293b']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                with st.expander("📄 Dettagli JSON"):
                                    st.json(stats)
                    except Exception as e:
                        st.error(f"Errore durante il preprocessing: {e}")

    with tab_fs:
        st.header("Fase 2: Feature Selection tramite QUBO")
        st.write("Ottimizza la scelta delle feature massimizzando la correlazione col target e minimizzandola tra le feature.")
        
        with st.expander("⚙️ Parametri Input/Output", expanded=False):
            norm_csv = st.text_input("Dataset normalizzato (input)", os.path.join(OUTPUT_DIR, "normalized.csv"))
            fs_target_col = st.text_input("Nome colonna Target", "target", key="fs_target")
            out_train = st.text_input("Output Training CSV", os.path.join(OUTPUT_DIR, "training_reduced.csv"))
            out_test = st.text_input("Output Test CSV", os.path.join(OUTPUT_DIR, "test_reduced.csv"))
            out_optim = st.text_input("Output QUBO optims CSV", os.path.join(OUTPUT_DIR, "optimizations.csv"))
            fs_out_json = st.text_input("Output statistiche JSON", os.path.join(OUTPUT_DIR, "feature_selection_result.json"))
        
        st.subheader("Parametri QUBO")
        col1, col2 = st.columns(2)
        with col1:
            perc_selected = st.slider("Percentuale di feature da selezionare", 0.05, 1.0, 0.20, step=0.01)
            perc_test = st.slider("Percentuale di test split", 0.1, 0.5, 0.30, step=0.01)
        with col2:
            allowance = st.number_input("Tolleranza numero feature (allowance)", min_value=0, value=1)
            alpha_comps = st.number_input("Max calcoli alpha", min_value=1, value=100)
            seed_fs = st.number_input("Seed per riproducibilità", value=42, key="seed_fs")

        can_run = os.path.exists(norm_csv)
        if not can_run:
            st.warning("⚠️ Esegui prima la Fase 1 per generare il dataset normalizzato.")

        if st.button("Avvia Feature Selection (QUBO)", type="primary", use_container_width=True, disabled=not can_run):
            with st.spinner("Calcolo Matrice Q e Ottimizzazione in corso..."):
                try:
                    select_features(
                        normalized_csv=norm_csv,
                        reducedTrain_csv=out_train,
                        reducedTest_csv=out_test,
                        output_ottim_csv=out_optim,
                        output_json=fs_out_json,
                        target_column=fs_target_col,
                        percTest=perc_test,
                        percSelected=perc_selected,
                        allowance=int(allowance),
                        seed=int(seed_fs),
                        alpha_computations=int(alpha_comps)
                    )
                    st.success("Feature Selection completata!")
                    scroll_to_bottom()
                    
                    stats = load_json(fs_out_json)
                    if stats:
                        with st.container(border=True):
                            st.subheader("🎯 Risultati Ottimizzazione")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Feature Originarie", stats.get("n_features", 0))
                            c2.metric("Feature Selezionate", stats.get("n_selected", 0))
                            c3.metric("Alpha Ottimo", f"{stats.get('alpha', 0):.4f}")
                            
                            st.write("**Feature Selezionate:**")
                            st.info(", ".join(stats.get("selected_feature_names", [])))
                            
                            if os.path.exists(out_optim):
                                df_opt = pd.read_csv(out_optim)
                                fig = px.line(df_opt, x=df_opt.index, y='alpha', title='Ricerca di Alpha Ottimale', labels={'index':'Iterazione'})
                                fig.update_traces(line_color='#00d4ff')
                                st.plotly_chart(fig, use_container_width=True)
                                
                            with st.expander("📄 Dettagli Statistiche JSON"):
                                st.json(stats)
                except Exception as e:
                    st.error(f"Errore: {e}")

    with tab_train:
        st.header("Fase 3: Learning del Classificatore")
        
        with st.expander("⚙️ Parametri Input/Output", expanded=False):
            train_csv = st.text_input("Training Dataset ridotto", os.path.join(OUTPUT_DIR, "training_reduced.csv"))
            tr_target_col = st.text_input("Nome colonna Target", "target", key="tr_target")
            out_model = st.text_input("Salvataggio Modello (.joblib)", os.path.join(OUTPUT_DIR, "model.joblib"))
            tr_out_json = st.text_input("Salvataggio Metriche", os.path.join(OUTPUT_DIR, "training_metrics.json"))
        
        classifier = st.radio("Algoritmo di Classificazione", ["random_forest", "gradient_boosting", "logistic_regression"], horizontal=True)
        seed_tr = st.number_input("Seed", value=42, key="seed_tr")

        can_run = os.path.exists(train_csv)
        if not can_run:
            st.warning("⚠️ Esegui prima la Fase 2 per generare il dataset di training.")

        if st.button("Addestra Modello", type="primary", use_container_width=True, disabled=not can_run):
            with st.spinner(f"Addestramento {classifier} in corso..."):
                try:
                    train(
                        classifier=classifier,
                        reducedTrain_csv=train_csv,
                        target_column=tr_target_col,
                        model_path=out_model,
                        metrics_json=tr_out_json,
                        seed=int(seed_tr)
                    )
                    st.success("Addestramento completato e modello salvato!")
                    scroll_to_bottom()
                    
                    stats = load_json(tr_out_json)
                    if stats:
                        with st.container(border=True):
                            st.subheader("🎯 Metriche di Addestramento")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Tempo di Addestramento", f"{stats.get('training_time', 0):.2f} s")
                            c2.metric("Campioni nel Training", stats.get("n_samples", 0))
                            c3.metric("Target=1 Ratio", f"{stats.get('target_1_percentage', 0):.1f}%")
                            
                            with st.expander("📄 Dettagli Modello"):
                                st.json(stats)
                except Exception as e:
                    st.error(f"Errore: {e}")

    with tab_pred:
        st.header("Fase 4: Classificazione e Predizione")
        
        with st.expander("⚙️ Parametri Input/Output", expanded=False):
            test_csv = st.text_input("Test Dataset", os.path.join(OUTPUT_DIR, "test_reduced.csv"))
            pr_target_col = st.text_input("Nome colonna Target", "target", key="pr_target")
            model_path = st.text_input("Modello addestrato (.joblib)", os.path.join(OUTPUT_DIR, "model.joblib"))
            out_preds = st.text_input("Output Predizioni CSV", os.path.join(OUTPUT_DIR, "predictions.csv"))
            out_stats = st.text_input("Output Statistiche Classificazione", os.path.join(OUTPUT_DIR, "classification_stats.json"))

        can_run = True
        if not os.path.exists(test_csv):
            st.warning("⚠️ Dataset di test mancante. Esegui la Fase 2.")
            can_run = False
        elif not os.path.exists(model_path):
            st.warning("⚠️ Modello mancante. Esegui la Fase 3.")
            can_run = False
        elif os.path.getmtime(model_path) < os.path.getmtime(test_csv):
            st.warning("⚠️ Attenzione: Dataset modificato dopo l'addestramento. Ri-addestra il modello (Fase 3).")

        if st.button("Genera Predizioni", type="primary", use_container_width=True, disabled=not can_run):
            with st.spinner("Predizione in corso..."):
                try:
                    predict(
                        reduced_Test_csv=test_csv,
                        target_column=pr_target_col,
                        model_path=model_path,
                        predictions_csv=out_preds,
                        classif_stats_json=out_stats
                    )
                    st.success("Predizioni generate con successo!")
                    scroll_to_bottom()
                    
                    stats = load_json(out_stats)
                    if stats:
                        with st.container(border=True):
                            st.subheader("🎯 Performance del Modello")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Accuracy", f"{stats.get('accuracy', 0):.4f}")
                            c2.metric("ROC AUC", f"{stats.get('roc_auc', 0):.4f}")
                            c3.metric("F1 Score (Cls 1)", f"{stats.get('class_1', {}).get('f1', 0):.4f}")
                            c4.metric("Precision (Cls 1)", f"{stats.get('class_1', {}).get('precision', 0):.4f}")
                            
                            metrics_df = pd.DataFrame({
                                'Metric': ['Precision', 'Recall', 'F1-Score'],
                                'Classe 0': [stats.get('class_0', {}).get('precision', 0), stats.get('class_0', {}).get('recall', 0), stats.get('class_0', {}).get('f1', 0)],
                                'Classe 1': [stats.get('class_1', {}).get('precision', 0), stats.get('class_1', {}).get('recall', 0), stats.get('class_1', {}).get('f1', 0)]
                            })
                            fig = px.bar(metrics_df, x='Metric', y=['Classe 0', 'Classe 1'], barmode='group', title="Metriche per Classe", color_discrete_sequence=['#1e293b', '#00d4ff'])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            with st.expander("📄 Dettagli Report Completo"):
                                st.json(stats)
                                
                            if os.path.exists(out_preds):
                                st.write("Anteprima file Predizioni:")
                                st.dataframe(pd.read_csv(out_preds).head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Errore: {e}")

if __name__ == "__main__":
    main()
