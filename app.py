import streamlit as st
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import AllChem
import numpy as np
from matplotlib import cm
from ochem import mycalc, predict_only
import pandas as pd
import warnings


warnings.filterwarnings("ignore")

# =========================
# MODEL FILES
# =========================

MODEL_FILE = 'fxr_svc_fcfp4.pkl'

sdf1 = "training_fp_minimized.sdf"
sdf2 = "training_tc_minimized.sdf"

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_model():

    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    return model


@st.cache_resource
def load_tc_model():

    with open("fatimaBest.pickle", "rb") as f:
        model = pickle.load(f)

    return model


# =========================
# LOAD TRAINING FINGERPRINTS
# =========================

@st.cache_resource
def load_training_fps(sdf_file):

    fps = []

    suppl = Chem.SDMolSupplier(sdf_file)

    for mol in suppl:

        if mol is not None:

            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=2,
                nBits=1024,
                useFeatures=True,
                useChirality=True
            )

            fps.append(fp)

    return fps


# =========================
# GENERATE FINGERPRINT
# =========================

def generate_fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=1024,
        useFeatures=True,
        useChirality=True
    )

    arr = np.zeros((1,), dtype=int)

    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

    return mol, arr


# =========================
# SIMILARITY MAP FUNCTIONS
# =========================

def fpFunction(m, atomId=-1):

    fp = SimilarityMaps.GetMorganFingerprint(
        m,
        atomId=atomId,
        radius=2,
        nBits=1024,
        useChirality=True
    )

    return fp


def getProba(fp, predictionFunction):

    return predictionFunction((fp,))[0][1]


# =========================
# APPLICABILITY DOMAIN
# =========================

def applicability(smiles, sdf):

    m = Chem.MolFromSmiles(smiles)

    if m is None:
        return 0.0

    query_fp = AllChem.GetMorganFingerprintAsBitVect(
        m,
        radius=2,
        nBits=1024,
        useFeatures=True,
        useChirality=True
    )

    train_fps = load_training_fps(sdf)

    similarities = []

    for fp in train_fps:

        sim = DataStructs.FingerprintSimilarity(fp, query_fp)

        similarities.append(sim)

    return max(similarities)


# =========================
# PLOT SIMILARITY MAP
# =========================

def plot_similarity_map(mol, model):

    d = Draw.MolDraw2DCairo(400, 400)

    SimilarityMaps.GetSimilarityMapForModel(
        mol,
        fpFunction,
        lambda x: getProba(x, model.predict_proba),
        draw2d=d
    )

    d.FinishDrawing()

    return d


# =========================
# STREAMLIT APP
# =========================

def main():

    st.title("PDE5A Calculator")

    st.subheader(
        "This app calculates the activity of chemical compounds "
        "against PDE5A using fingerprint-based and "
        "Transformer-CNN-based models"
    )

    st.write(
        "Input a SMILES notation of a chemical compound "
        "to predict its activity."
    )

    # =========================
    # INPUT
    # =========================

    smiles = st.text_input(
        "Enter SMILES Notation:",
        ""
    )

    if smiles:

        mol, fingerprint = generate_fingerprint(smiles)

        if mol is None:

            st.error(
                "Invalid SMILES notation. Please try again."
            )

        else:

            # =========================
            # LOAD MODELS
            # =========================

            model = load_model()

            tc_model = load_tc_model()

            # =========================================================
            # FINGERPRINT MODEL
            # =========================================================

            st.write(
                "## Fingerprint-Based Model"
            )

            prediction = model.predict([fingerprint])

            activity = (
                "Active (IC50 < 1000 nM)"
                if prediction[0] == 1
                else "Inactive (IC50 >= 1000 nM)"
            )

            st.write(
                f"Predicted Activity: **{activity}**"
            )

            # =========================
            # APPLICABILITY DOMAIN
            # =========================

            value = applicability(smiles, sdf1)

            if value >= 0.4:

                st.write(
                    "The compound falls within the "
                    "applicability domain of the "
                    "fingerprint-based model"
                )

            else:

                st.write(
                    "The compound falls outside the "
                    "applicability domain of the "
                    "fingerprint-based model"
                )

            # =========================
            # SIMILARITY MAP
            # =========================

            st.write(
                "### Similarity Map"
            )

            res = plot_similarity_map(mol, model)

            fig = res.GetDrawingText()

            st.image(fig)

            st.markdown("**Colour scheme:**")

            st.markdown(
                '<span style="color:green">'
                'Fragments increasing inhibitory activity'
                '</span>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<span style="color:red">'
                'Fragments decreasing inhibitory activity'
                '</span>',
                unsafe_allow_html=True
            )

            # =========================================================
            # TRANSFORMER-CNN MODEL
            # =========================================================

            st.write(
                "## Transformer-CNN-Based Model"
            )

            with st.spinner(
                "Running Transformer-CNN prediction..."
            ):

                act_trans = predict_only(smiles)

            activity = (
                "Active (IC50 < 1000 nM)"
                if act_trans == 'Active'
                else "Inactive (IC50 >= 1000 nM)"
            )

            st.write(
                f"Predicted Activity: **{activity}**"
            )

            # =========================
            # APPLICABILITY DOMAIN
            # =========================

            value = applicability(smiles, sdf2)

            if value >= 0.4:

                st.write(
                    "The compound falls within the "
                    "applicability domain of the "
                    "Transformer-CNN model"
                )

            else:

                st.write(
                    "The compound falls outside the "
                    "applicability domain of the "
                    "Transformer-CNN model"
                )

            # =========================================================
            # OPTIONAL LRP INTERPRETATION
            # =========================================================

            st.write(
                "## Optional Transformer-CNN Interpretation"
            )

            num_atoms = mol.GetNumAtoms()

            if num_atoms > 50:

                st.warning(
                    "Large molecule detected. "
                    "LRP interpretation may take several minutes."
                )

            if st.button(
                "Generate Transformer Interpretation Map (slow for large molecules," 
                "may be used from Stremlit app from Github link given below)"
            ):

                with st.spinner(
                    "Running Transformer-CNN interpretation..."
                ):

                    act_trans, fig2, l1, l2 = mycalc(
                        tc_model,
                        smiles
                    )

                st.write(
                    str(l2 - l1)
                    + ' out of '
                    + str(l2)
                    + ' atoms were successfully propagated '
                    + 'to position-wise layers'
                )

                st.image(fig2)

                st.markdown("**Colour scheme:**")

                st.markdown(
                    '<span style="color:green">'
                    'Fragments increasing inhibitory activity'
                    '</span>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<span style="color:red">'
                    'Fragments decreasing inhibitory activity'
                    '</span>',
                    unsafe_allow_html=True
                )


# =========================
# FOOTER
# =========================

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
    }
    </style>

    <div class="footer">
        Made using Streamlit by
        Dr. Amit Kumar Halder,
        Post Doctoral Researcher,
        LAQV/REQUIMTE,
        University of Porto, Portugal |
        <a href="https://laqv.requimte.pt/" target="_blank">
        About Us
        </a>
        | Github link for this Streamlit app:
        <a href="https://github.com/ncordeirfcup/PDE5A_Streamlit" target="_blank">
        Click here
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    main()