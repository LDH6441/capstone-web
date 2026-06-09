"""
청년 마음 건강 체크인 - Streamlit 버전 (초고속 배포용)
- Flask 뼈대를 걷어내고 Streamlit 전용 UI로 완벽 개조
"""
import os
import json
import traceback
import numpy as np
import pandas as pd
import streamlit as st

# Gemini API
from google import genai
from google.genai import types as genai_types

# ==========================================
# 1. 페이지 및 기본 설정
# ==========================================
st.set_page_config(page_title="청년 마음 건강 체크인", page_icon="", layout="wide")

# Gemini API 설정
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyD_QpV-qtEpPtfabD49NbHh_j7jXOGNSqM')
GEMINI_MODEL_NAME = 'gemini-2.5-flash'

@st.cache_resource
def get_gemini_client():
    if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE':
        try:
            return genai.Client(api_key=GEMINI_API_KEY)
        except:
            return None
    return None

GEMINI_CLIENT = get_gemini_client()

# ==========================================
# 2. AI 모델 로드 (캐싱하여 속도 향상)
# ==========================================
@st.cache_resource
def load_ai_model():
    MODEL_PATH = './models/best_pycaret_model'
    try:
        from pycaret.classification import load_model
        ARTIFACTS = load_model(MODEL_PATH, verbose=False)
        return ARTIFACTS
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

ARTIFACTS = load_ai_model()

# ==========================================
# 3. 데이터 전처리 로직 (기존 함수 100% 유지)
# ==========================================
MISSING_CODES = {
    'BP16_11': [88, 99], 'BP16_21': [88, 99], 'BE8_1': [88, 99], 'BP16_2': [88, 99],
    'LQ_4EQL': [8, 9], 'sex': [9], 'LQ4_00': [8, 9], 'BM7': [8, 9], 'BM8': [8, 9],
    'BE5_1': [8, 9], 'BO1': [8, 9], 'BO1_1': [8, 9], 'BO1_2': [8, 9], 'BO1_3': [8, 9],
    'BS13': [8, 9], 'BS1_1': [8, 9], 'BD1': [8, 9], 'BD7_4': [8, 9], 'LQ1_sb': [8, 9], 'BP6_2': [8, 9],
}

def calculate_ainc(payload):
    unit, amount = payload.get('ainc_unit1'), payload.get('ainc_1')
    if unit is None or amount is None: return np.nan
    try:
        unit, amount = int(unit), float(amount)
        if unit == 1: return round(amount / 12, 2)
        elif unit == 2: return amount
    except: pass
    return np.nan

def replace_missing(value, codes):
    if value is None: return np.nan
    try: v = float(value)
    except: return np.nan
    if codes and v in codes: return np.nan
    return v

def compute_derived(df):
    def normalize_bedtime(h): return h + 24 if pd.notna(h) and 1 <= h <= 12 else h
    df['bed_wk_norm']   = df['BP16_11'].apply(normalize_bedtime)
    df['bed_wd_norm']   = df['BP16_21'].apply(normalize_bedtime)
    df['social_jetlag'] = df['bed_wd_norm'] - df['bed_wk_norm']
    df['sleep_short']   = (df['BP16_2'] < 6).astype(float)
    df['low_activity']  = (df['BE5_1'] <= 2).astype(float)
    df['risk_composite'] = (
        (df['BP16_2'] < 6).fillna(0).astype(int) +
        (df['BE5_1'] <= 2).fillna(0).astype(int) +
        (df['BS1_1'] != 3).fillna(0).astype(int) +
        (df['LQ_4EQL'] >= 2).fillna(0).astype(int) +
        (df['BO1'] >= 4).fillna(0).astype(int)
    ).astype(float)
    return df

def build_input_df(payload, artifacts):
    row = {'ainc': calculate_ainc(payload)}
    raw_input_cols = [
        'LQ_4EQL', 'sex', 'mt_nontrt', 'LQ4_00', 'BM7', 'BE5_1', 'BO1_1', 'BS13', 'BE8_1', 'BD7_4',
        'BO1', 'BS1_1', 'LQ1_sb', 'BP16_11', 'BP6_2', 'BP16_21', 'region', 'BM8', 'BO1_2', 'BO1_3',
        'BP16_2', 'BD1',
    ]
    for col in raw_input_cols:
        row[col] = replace_missing(payload.get(col), MISSING_CODES.get(col, []))
    
    df = pd.DataFrame([row])
    df = compute_derived(df)

    for col in artifacts['feature_cols_raw']:
        if col not in df.columns: df[col] = np.nan
    df = df[artifacts['feature_cols_raw']].copy()

    cont_exist = [c for c in artifacts['continuous_cols'] if c in df.columns]
    cat_exist  = [c for c in artifacts['categorical_cols'] if c in df.columns]

    if artifacts.get('imp_cont') and cont_exist: df[cont_exist] = artifacts['imp_cont'].transform(df[cont_exist])
    if artifacts.get('imp_cat') and cat_exist: df[cat_exist] = artifacts['imp_cat'].transform(df[cat_exist])
    if artifacts.get('scaled', True) and artifacts.get('scaler') and cont_exist:
        df[cont_exist] = artifacts['scaler'].transform(df[cont_exist])

    for col in cat_exist:
        levels = artifacts['category_levels'].get(col)
        if levels: df[col] = pd.Categorical(df[col], categories=levels)

    df = pd.get_dummies(df, columns=cat_exist, drop_first=False)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    df = df.reindex(columns=artifacts['feature_cols_onehot'], fill_value=0)
    return df

def predict(payload, artifacts):
    X = build_input_df(payload, artifacts)
    probs = artifacts['model'].predict_proba(X)[0]
    pred_class = int(np.argmax(probs))
    return pred_class, {str(i): float(p) for i, p in enumerate(probs)}, X

def compute_top_reasons(X, pred_class, payload, artifacts, top_k=3):
    import shap
    explainer = shap.TreeExplainer(artifacts['explainer_model'])
    shap_raw = explainer.shap_values(X, check_additivity=False)
    
    if isinstance(shap_raw, list): shap_values = np.stack(shap_raw, axis=-1)
    elif shap_raw.ndim == 3 and shap_raw.shape[1] == 3: shap_values = shap_raw.transpose(0, 2, 1)
    else: shap_values = shap_raw

    shap_for_pred = shap_values[0, :, pred_class]
    raw_features = set(artifacts['feature_cols_raw'])
    
    def map_to_raw(col):
        if col in raw_features: return col
        parts = col.rsplit('_', 1)
        if len(parts) == 2 and parts[0] in raw_features: return parts[0]
        return col

    contribution = {}
    for i, col in enumerate(artifacts['feature_cols_onehot']):
        base = map_to_raw(col)
        contribution[base] = contribution.get(base, 0.0) + float(shap_for_pred[i])

    sorted_feats = sorted(contribution.items(), key=lambda x: abs(x[1]), reverse=True)
    label_names = {int(k): str(v) for k, v in artifacts.get('label_names', {0: '정상', 1: '경도', 2: '중등도이상'}).items()}
    pred_label = label_names.get(pred_class, '')

    results = []
    for feature_name, contrib in sorted_feats[:top_k]:
        kor_name = artifacts.get('col_rename', {}).get(feature_name, feature_name)
        effect = f"'{pred_label}' 위험 증가" if contrib > 0 else f"보호 요인"
        results.append(f"- **{kor_name}** (기여도: {round(contrib, 4)}) : {effect}")
    return results

# ==========================================
# 4. Streamlit UI (화면 구성)
# ==========================================
st.title(" 청년 마음 건강 체크인 AI")

if not ARTIFACTS:
    st.error("서버에 모델이 로드되지 않았습니다. 관리자에게 문의하세요.")
    st.stop()

# 탭으로 화면 분리
tab1, tab2 = st.tabs(["📋 설문 검사하기", "💬 마루와 상담하기"])

with tab1:
    st.header("나의 상태 입력하기")
    st.info("정확한 분석을 위해 각 항목의 숫자를 입력해주세요. (결측치는 비워두거나 0 입력)")
    
    with st.form("survey_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_name = st.text_input("이름 (닉네임)", value="회원")
            ainc_unit1 = st.number_input("소득 단위 (1:연, 2:월)", value=2)
            ainc_1 = st.number_input("소득 금액", value=200)
            LQ_4EQL = st.number_input("LQ_4EQL", value=1)
            sex = st.number_input("성별 (sex)", value=1)
            mt_nontrt = st.number_input("치료경험 (mt_nontrt)", value=0)
            LQ4_00 = st.number_input("LQ4_00", value=1)
            BM7 = st.number_input("BM7", value=1)
            
        with col2:
            BE5_1 = st.number_input("BE5_1", value=1)
            BO1_1 = st.number_input("BO1_1", value=1)
            BS13 = st.number_input("BS13", value=1)
            BE8_1 = st.number_input("BE8_1", value=8)
            BD7_4 = st.number_input("BD7_4", value=1)
            BO1 = st.number_input("BO1", value=1)
            BS1_1 = st.number_input("BS1_1", value=3)
            LQ1_sb = st.number_input("LQ1_sb", value=1)
            
        with col3:
            BP16_11 = st.number_input("주중 취침시간 (BP16_11)", value=23)
            BP6_2 = st.number_input("BP6_2", value=0)
            BP16_21 = st.number_input("주말 취침시간 (BP16_21)", value=24)
            region = st.number_input("지역 (region)", value=1)
            BM8 = st.number_input("BM8", value=1)
            BO1_2 = st.number_input("BO1_2", value=1)
            BO1_3 = st.number_input("BO1_3", value=1)
            BP16_2 = st.number_input("수면시간 (BP16_2)", value=7)
            BD1 = st.number_input("BD1", value=1)

        submitted = st.form_submit_button("AI 분석 결과 확인", use_container_width=True)

    if submitted:
        # 데이터 묶기
        payload = {
            "ainc_unit1": ainc_unit1, "ainc_1": ainc_1, "LQ_4EQL": LQ_4EQL, "sex": sex,
            "mt_nontrt": mt_nontrt, "LQ4_00": LQ4_00, "BM7": BM7, "BE5_1": BE5_1,
            "BO1_1": BO1_1, "BS13": BS13, "BE8_1": BE8_1, "BD7_4": BD7_4, "BO1": BO1,
            "BS1_1": BS1_1, "LQ1_sb": LQ1_sb, "BP16_11": BP16_11, "BP6_2": BP6_2,
            "BP16_21": BP16_21, "region": region, "BM8": BM8, "BO1_2": BO1_2,
            "BO1_3": BO1_3, "BP16_2": BP16_2, "BD1": BD1
        }
        
        with st.spinner("AI가 분석 중입니다..."):
            try:
                pred_class, proba, X = predict(payload, ARTIFACTS)
                label_names = {int(k): str(v) for k, v in ARTIFACTS.get('label_names', {0: '정상', 1: '경도', 2: '중등도이상'}).items()}
                pred_label = label_names.get(pred_class, '알 수 없음')
                
                # 결과 출력
                st.success(f"### 분석 완료! 결과: {pred_label}")
                st.write(f"정상: {proba['0']*100:.1f}% | 경도 우울: {proba['1']*100:.1f}% | 중등도 우울: {proba['2']*100:.1f}%")
                
                if ARTIFACTS.get('explainer_model'):
                    st.subheader("🔍 주요 원인 분석 (상위 3개)")
                    reasons = compute_top_reasons(X, pred_class, payload, ARTIFACTS)
                    for r in reasons:
                        st.write(r)
                
                # 챗봇용 데이터 저장 (세션)
                st.session_state['user_name'] = user_name
                st.session_state['pred_label'] = pred_label
                st.session_state['pred_class'] = pred_class
                st.session_state['proba'] = proba
                
                st.info("👉 상단의 [💬 마루와 상담하기] 탭으로 이동해서 AI 동반자와 대화를 나눠보세요!")
                
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")

with tab2:
    st.header("💬 AI 동반자 '마루'와 대화하기")
    
    if 'pred_label' not in st.session_state:
        st.warning("먼저 [📋 설문 검사하기] 탭에서 검사를 완료해주세요!")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 기존 대화 출력
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 사용자 입력 처리
        if prompt := st.chat_input("여기에 대화를 입력하세요..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            if GEMINI_CLIENT is None:
                st.error("Gemini API가 설정되지 않았습니다.")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("마루가 생각 중입니다..."):
                        # 프롬프트 생성
                        name = st.session_state.get('user_name', '회원')
                        label = st.session_state.get('pred_label', '')
                        system_prompt = f"""당신은 '마루'라는 이름의 따뜻한 마음 건강 동반자입니다.
                        사용자 이름: {name}, 검사 결과: {label}
                        - 감정에 공감하고 짧고 따뜻하게 응답하세요.
                        - 의료 진단을 내리거나 처방을 하지 마세요.
                        - 자살/자해 키워드 감지 시 109 또는 1577-0199를 부드럽게 안내하세요.
                        """
                        
                        contents = [genai_types.Content(role='user' if m["role"] == 'user' else 'model', 
                                                        parts=[genai_types.Part.from_text(text=m["content"])]) 
                                    for m in st.session_state.messages[:-1]]
                        contents.append(genai_types.Content(role='user', parts=[genai_types.Part.from_text(text=prompt)]))
                        
                        try:
                            response = GEMINI_CLIENT.models.generate_content(
                                model=GEMINI_MODEL_NAME,
                                contents=contents,
                                config=genai_types.GenerateContentConfig(
                                    system_instruction=system_prompt,
                                    temperature=0.7, max_output_tokens=400
                                )
                            )
                            reply = response.text
                        except Exception as e:
                            reply = f"오류가 발생했어요. 다시 시도해주세요. ({e})"
                            
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
