"""
청년 마음 건강 체크인 - Flask 백엔드
- 신버전(app2.py: Gemini 챗봇, PyCaret load_model) + 구버전(app.py: 상위 3개 원인 SHAP 설명) 병합

- best_pycaret_model.pkl 안에 dict로 저장된 model + 전처리 객체를 직접 사용
- dict 구조:
    'model'              : LGBMClassifier (학습된 LightGBM)
    'thresholds'         : ndarray (클래스별 결정 임계값)
    'imp_cont'           : SimpleImputer (연속형 결측치 대체)
    'imp_cat'            : SimpleImputer (범주형 결측치 대체)
    'scaler'             : MinMaxScaler (연속형 스케일링)
    'feature_cols_raw'   : list (원본 입력 컬럼 23개)
    'continuous_cols'    : list (연속형 컬럼)
    'categorical_cols'   : list (범주형 컬럼)
    'category_levels'    : dict (각 범주형이 본 카테고리 값)
    'feature_cols_onehot': list (원-핫 후 75개 컬럼 순서)
    'label_names'        : dict (클래스 이름 매핑)
    'col_rename'         : dict (한국어 컬럼명 매핑)   ★ 구버전에서 사용
    'explainer_model'    : XGBClassifier 등 SHAP 가능 모델  ★ 구버전에서 사용
"""

import os
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# Gemini API (새 SDK: google-genai)
from google import genai
from google.genai import types as genai_types

# ════════════════════════════════════════════════════════
# Flask 앱 설정
# ════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# ════════════════════════════════════════════════════════
# Gemini 챗봇 설정
# ════════════════════════════════════════════════════════
# 환경변수 GEMINI_API_KEY 우선 사용, 없으면 아래 직접 입력값 사용
# 본인의 Gemini API 키를 입력하세요 (https://aistudio.google.com/apikey)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyD_QpV-qtEpPtfabD49NbHh_j7jXOGNSqM')

# 사용할 모델 — 무료 등급에서 잘 작동하는 모델
GEMINI_MODEL_NAME = 'gemini-2.5-flash'

GEMINI_CLIENT = None
try:
    if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE':
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        print(f"[INIT] ✓ Gemini 챗봇 준비 완료 (모델: {GEMINI_MODEL_NAME})")
    else:
        print("[INIT] ⚠ GEMINI_API_KEY가 설정되지 않아 챗봇은 비활성화됩니다.")
except Exception as e:
    print(f"[INIT] ✗ Gemini 초기화 실패: {e}")


# 챗봇 시스템 프롬프트 (마음이의 정체성)
def build_system_prompt(user_name, pred_label, pred_class, proba):
    name = user_name or '회원'
    proba_str = ''
    if proba:
        try:
            p = json.loads(proba) if isinstance(proba, str) else proba
            proba_str = (
                f"   - 정상 가능성: {float(p.get('0', 0)) * 100:.0f}%\n"
                f"   - 경도 우울 가능성: {float(p.get('1', 0)) * 100:.0f}%\n"
                f"   - 중등도 이상 가능성: {float(p.get('2', 0)) * 100:.0f}%"
            )
        except Exception:
            proba_str = ''

    result_block = ''
    if pred_label:
        result_block = f"""
[사용자의 마음 건강 검사 결과]
- 이름: {name}님
- AI 예측 결과: {pred_label}
{proba_str}
"""

    return f"""당신은 '마루'라는 이름의 따뜻한 마음 건강 동반자입니다.
{name}님과 한국어로 대화합니다.

【당신의 역할】
- 사용자의 감정을 공감하고 경청하기
- 짧고 따뜻하게 응답하기 (2-4문장 정도)
- 작은 자기돌봄 방법(수면, 산책, 호흡, 일기 등) 부드럽게 제안하기
- 사용자가 자신의 마음을 표현하도록 열린 질문 던지기

【당신의 정체성】
- 이름을 물어보면 "마루"라고 답하세요
- "AI냐?"라고 물으면 솔직하게 "네, 저는 AI 동반자 마루예요"라고 답하세요
- 의료진이나 상담사를 사칭하지 마세요

【절대 하지 말 것】
- 의료 진단을 내리지 마세요. "당신은 우울증입니다" 같은 단정 금지
- 약물 처방이나 의학적 조언 금지
- "그건 별거 아니에요" 같은 감정 무시 표현 금지
- 과한 위로나 비현실적인 낙관 ("다 잘 될 거예요!") 금지
- 사용자가 검사 결과를 부정적으로 받아들여도 결과를 강조하지 마세요

【위기 신호 감지 시】
사용자가 자살, 자해, 죽음, "사라지고 싶다" 등을 언급하면:
1. 먼저 공감하고 안전을 최우선으로 표현
2. 자살예방 상담전화 ☎ 109 (24시간) 또는 정신건강 위기상담 ☎ 1577-0199를 부드럽게 안내
3. "지금 당장 도움을 청하셔도 괜찮아요"라고 알리기

【응답 톤】
- 친근하지만 가볍지 않게
- 이모지는 한 응답에 최대 1개까지만, 자연스러울 때만
- "~요" 체로 부드럽게
- {name}님이라고 가끔 이름을 불러주기
{result_block}
이 정보를 바탕으로 따뜻한 대화를 시작해주세요.
사용자가 결과에 대해 묻거나 자신의 감정을 이야기하면 공감하며 응답하세요.
"""


# 위기 키워드 감지
CRISIS_KEYWORDS = [
    '자살', '죽고 싶', '죽고싶', '자해', '뛰어내', '목매', '극단적 선택',
    '없어지고 싶', '사라지고 싶', '살기 싫', '살기싫',
]


def detect_crisis(text):
    if not text:
        return False
    return any(kw in text for kw in CRISIS_KEYWORDS)


# ════════════════════════════════════════════════════════
# DB 모델
# ════════════════════════════════════════════════════════
class Survey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(50))
    user_age = db.Column(db.Integer)
    payload = db.Column(db.Text(), nullable=False)
    pred_class = db.Column(db.Integer, nullable=False)
    pred_label = db.Column(db.String(50), nullable=False)
    proba = db.Column(db.Text())
    create_date = db.Column(db.DateTime, nullable=False, default=datetime.now)


with app.app_context():
    db.create_all()


# ════════════════════════════════════════════════════════
# 모델 + 전처리 객체 로드 (PyCaret의 load_model 사용)
# - PyCaret이 저장한 형식이라 PyCaret으로 열어야 dict가 그대로 풀림
# ════════════════════════════════════════════════════════
MODEL_PATH = './models/best_pycaret_model'  # 확장자 없이 경로만

ARTIFACTS = None
MODEL = None
SCALER = None
SCALED = True
IMP_CONT = None
IMP_CAT = None
FEATURE_COLS_RAW = None
CONTINUOUS_COLS = None
CATEGORICAL_COLS = None
CATEGORY_LEVELS = None
FEATURE_COLS_ONEHOT = None
LABEL_NAMES = None
THRESHOLDS = None
COL_RENAME = {}              # ★ 구버전 기능 복원: 한국어 컬럼명
EXPLAINER_MODEL = None       # ★ 구버전 기능 복원: SHAP 설명용 모델

try:
    from pycaret.classification import load_model
    print(f"[INIT] 모델 + 전처리 아티팩트 로드 중: {MODEL_PATH}.pkl")
    ARTIFACTS = load_model(MODEL_PATH, verbose=False)

    if not isinstance(ARTIFACTS, dict):
        raise ValueError(f"예상 타입이 dict인데 실제는 {type(ARTIFACTS).__name__}")

    MODEL               = ARTIFACTS['model']
    SCALER              = ARTIFACTS.get('scaler')
    SCALED              = ARTIFACTS.get('scaled', True)  # 기본 True (이전 모델 호환)
    IMP_CONT            = ARTIFACTS.get('imp_cont')
    IMP_CAT             = ARTIFACTS.get('imp_cat')
    FEATURE_COLS_RAW    = ARTIFACTS['feature_cols_raw']
    CONTINUOUS_COLS     = ARTIFACTS['continuous_cols']
    CATEGORICAL_COLS    = ARTIFACTS['categorical_cols']
    CATEGORY_LEVELS     = ARTIFACTS['category_levels']
    FEATURE_COLS_ONEHOT = ARTIFACTS['feature_cols_onehot']
    LABEL_NAMES         = ARTIFACTS.get('label_names', {0: '정상', 1: '경도', 2: '중등도이상'})
    THRESHOLDS          = ARTIFACTS.get('thresholds')
    COL_RENAME          = ARTIFACTS.get('col_rename', {})           # ★ 구버전 기능 복원
    EXPLAINER_MODEL     = ARTIFACTS.get('explainer_model')          # ★ 구버전 기능 복원

    # int 키로 변환 (JSON 호환)
    LABEL_NAMES = {int(k): str(v) for k, v in LABEL_NAMES.items()}

    print(f"[INIT] ✓ 로드 완료")
    print(f"[INIT]   model            : {type(MODEL).__name__}")
    print(f"[INIT]   원본 입력 컬럼   : {len(FEATURE_COLS_RAW)}개")
    print(f"[INIT]   연속형           : {len(CONTINUOUS_COLS)}개 {CONTINUOUS_COLS}")
    print(f"[INIT]   범주형           : {len(CATEGORICAL_COLS)}개")
    print(f"[INIT]   원-핫 후 컬럼    : {len(FEATURE_COLS_ONEHOT)}개")
    print(f"[INIT]   스케일링 사용    : {SCALED}")
    print(f"[INIT]   클래스 이름      : {LABEL_NAMES}")
    print(f"[INIT]   explainer_model  : {'있음 ✓' if EXPLAINER_MODEL else '없음 ✗'}")
    if THRESHOLDS is not None:
        print(f"[INIT]   thresholds       : {THRESHOLDS}")

except Exception as e:
    print(f"[INIT] ✗ 로드 실패: {e}")
    traceback.print_exc()


# ════════════════════════════════════════════════════════
# 결측 코드 매핑 (학습 코드의 replace_missing_codes 규칙)
# - 변수의 최대값에 따라 결측 코드가 다름
# ════════════════════════════════════════════════════════
MISSING_CODES = {
    # 2자리 변수 (max 24시간 등)
    'BP16_11': [88, 99],
    'BP16_21': [88, 99],
    'BE8_1':   [88, 99],
    'BP16_2':  [88, 99],
    # 1자리 변수 - 학습 코드에서 9가 30% 미만인 경우 결측 처리
    'LQ_4EQL': [8, 9],
    'sex':     [9],
    'LQ4_00':  [8, 9],
    'BM7':     [8, 9],
    'BM8':     [8, 9],
    'BE5_1':   [8, 9],
    'BO1':     [8, 9],
    'BO1_1':   [8, 9],
    'BO1_2':   [8, 9],
    'BO1_3':   [8, 9],
    'BS13':    [8, 9],
    'BS1_1':   [8, 9],
    'BD1':     [8, 9],
    'BD7_4':   [8, 9],
    'LQ1_sb':  [8, 9],
    'BP6_2':   [8, 9],
    # mt_nontrt: 0/1 코딩, 결측 코드 없음 (미응답 시 NaN)
    # region: 1~17, 결측 코드 없음
    # ainc: 연속형, 결측 코드 없음 (계산 실패 시 NaN)
}


# ════════════════════════════════════════════════════════
# 전처리: payload → 원-핫 인코딩된 75컬럼 DataFrame
# ════════════════════════════════════════════════════════
def calculate_ainc(payload):
    """ainc_unit1 + ainc_1로부터 월평균 가구총소득 계산"""
    unit = payload.get('ainc_unit1')
    amount = payload.get('ainc_1')

    if unit is None or amount is None:
        return np.nan
    try:
        unit = int(unit)
        amount = float(amount)
    except (ValueError, TypeError):
        return np.nan

    if unit == 1:    # 연 → 월
        return round(amount / 12, 2)
    elif unit == 2:  # 월
        return amount
    return np.nan


def replace_missing(value, codes):
    """결측 코드를 NaN으로 변환"""
    if value is None:
        return np.nan
    try:
        v = float(value)
    except (ValueError, TypeError):
        return np.nan
    if codes and v in codes:
        return np.nan
    return v


def compute_derived(df):
    """
    학습 때와 동일한 파생변수 계산 (팀원이 제공한 inference_template.py와 동일)
    절대 수정 X — 학습 시 사용된 공식 그대로
    """
    def normalize_bedtime(h):
        if pd.isna(h):
            return np.nan
        return h + 24 if 1 <= h <= 12 else h

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


def build_input_df(payload):
    """
    프론트엔드 payload → 학습 시와 동일한 전처리 적용 → 모델 입력 DataFrame

    팀원이 제공한 inference_template.py의 predict() 함수 흐름을 따름:
    1) 원본 변수 입력 → DataFrame
    2) 파생변수 계산 (compute_derived)
    3) feature_cols_raw 순서로 재정렬 (없는 컬럼은 NaN)
    4) imputer로 결측치 대체
    5) scaler로 스케일링
    6) 원-핫 인코딩
    7) feature_cols_onehot 순서로 reindex
    """
    # 1) 원본 사용자 입력값 1행 DataFrame 만들기
    #    프론트엔드에서 보내는 원본 변수들 + ainc 계산
    row = {}

    # ainc는 ainc_unit1 + ainc_1로 계산
    row['ainc'] = calculate_ainc(payload)

    # 나머지 원본 변수: 결측 코드 → NaN 변환
    raw_input_cols = [
        'LQ_4EQL', 'sex', 'mt_nontrt', 'LQ4_00', 'BM7',
        'BE5_1', 'BO1_1', 'BS13', 'BE8_1', 'BD7_4',
        'BO1', 'BS1_1', 'LQ1_sb', 'BP16_11', 'BP6_2',
        'BP16_21', 'region', 'BM8', 'BO1_2', 'BO1_3',
        'BP16_2', 'BD1',
    ]
    for col in raw_input_cols:
        raw = payload.get(col)
        codes = MISSING_CODES.get(col, [])
        row[col] = replace_missing(raw, codes)

    df = pd.DataFrame([row])

    # 2) ★ 파생변수 계산 (핵심!)
    df = compute_derived(df)

    # 3) 학습 모델이 기대하는 컬럼 순서로 재정렬 (없는 컬럼은 NaN)
    for col in FEATURE_COLS_RAW:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FEATURE_COLS_RAW].copy()

    # 4) 결측치 대체 (학습 시 fit된 imputer로 transform만)
    cont_exist = [c for c in CONTINUOUS_COLS if c in df.columns]
    cat_exist  = [c for c in CATEGORICAL_COLS if c in df.columns]

    if IMP_CONT is not None and cont_exist:
        df[cont_exist] = IMP_CONT.transform(df[cont_exist])
    if IMP_CAT is not None and cat_exist:
        df[cat_exist] = IMP_CAT.transform(df[cat_exist])

    # 5) 스케일링 (연속형, MinMaxScaler) — 학습 시 사용했을 때만 적용
    if SCALED and SCALER is not None and cont_exist:
        df[cont_exist] = SCALER.transform(df[cont_exist])

    # 6) 원-핫 인코딩 (학습 때와 같은 카테고리 레벨 보장)
    for col in cat_exist:
        levels = CATEGORY_LEVELS.get(col)
        if levels:
            df[col] = pd.Categorical(df[col], categories=levels)

    df = pd.get_dummies(df, columns=cat_exist, drop_first=False)

    # bool → int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 7) feature_cols_onehot 순서대로 reindex (없는 컬럼은 0으로)
    df = df.reindex(columns=FEATURE_COLS_ONEHOT, fill_value=0)

    return df


# ════════════════════════════════════════════════════════
# 예측
# ════════════════════════════════════════════════════════
def predict(payload):
    X = build_input_df(payload)

    # 확률 예측
    probs = MODEL.predict_proba(X)[0]  # shape: (3,)

    # 클래스 결정: 단순 argmax (확률이 가장 높은 클래스 선택)
    # - 이전엔 thresholds 보정을 했으나, 중등도이상 임계값(0.21)이 너무 낮아
    #   확률이 정상 클래스보다 낮아도 무조건 중등도로 분류되는 문제 발생
    # - 모델 본연의 판단을 신뢰하는 표준 방식으로 변경
    pred_class = int(np.argmax(probs))

    proba_dict = {str(i): float(p) for i, p in enumerate(probs)}
    return pred_class, proba_dict, X


# ════════════════════════════════════════════════════════
# ★ 구버전 기능 복원: 상위 K개 기여 변수 (SHAP 기반)
# ════════════════════════════════════════════════════════
def compute_top_reasons(X, pred_class, payload, top_k=3):
    """예측된 클래스에 가장 큰 영향을 준 상위 K개 변수 (SHAP 기반)"""
    import shap

    explainer = shap.TreeExplainer(EXPLAINER_MODEL)
    shap_raw = explainer.shap_values(X, check_additivity=False)

    # shape 통일: (samples, features, classes)
    if isinstance(shap_raw, list):
        shap_values = np.stack(shap_raw, axis=-1)
    elif shap_raw.ndim == 3 and shap_raw.shape[1] == 3:
        shap_values = shap_raw.transpose(0, 2, 1)
    else:
        shap_values = shap_raw

    shap_for_pred = shap_values[0, :, pred_class]

    # 원-핫 → 원본 변수 집계
    raw_features = set(FEATURE_COLS_RAW)
    def map_to_raw(col):
        if col in raw_features:
            return col
        parts = col.rsplit('_', 1)
        if len(parts) == 2 and parts[0] in raw_features:
            return parts[0]
        return col

    contribution = {}
    for i, col in enumerate(FEATURE_COLS_ONEHOT):
        base = map_to_raw(col)
        contribution[base] = contribution.get(base, 0.0) + float(shap_for_pred[i])

    sorted_feats = sorted(contribution.items(),
                          key=lambda x: abs(x[1]), reverse=True)
    pred_label = LABEL_NAMES.get(pred_class, '')

    results = []
    for feature_name, contrib in sorted_feats[:top_k]:
        kor_name = COL_RENAME.get(feature_name, feature_name)
        raw_value = payload.get(feature_name)
        if contrib > 0:
            effect = f"'{pred_label}'로 분류되는 데 기여 (위험 증가)"
        else:
            effect = f"'{pred_label}'와 멀어지는 방향 (보호 요인)"
        results.append({
            'feature'     : feature_name,
            'feature_kor' : kor_name,
            'value'       : raw_value,
            'contribution': round(contrib, 4),
            'effect'      : effect,
        })
    return results


def explain_prediction(payload, top_k=3):
    """
    예측에 가장 큰 영향을 준 상위 K개 변수 반환
    Returns: (pred_class, proba_dict, top_reasons)
    """
    if EXPLAINER_MODEL is None:
        raise RuntimeError("explainer_model이 번들에 없음 — 학습 노트북에서 추가 후 재저장 필요")

    # 1) 예측 (기존 predict 재사용)
    pred_class, proba_dict, X = predict(payload)

    # 2) 상위 이유 계산
    top_reasons = compute_top_reasons(X, pred_class, payload, top_k=top_k)

    return pred_class, proba_dict, top_reasons


# ════════════════════════════════════════════════════════
# 라우트
# ════════════════════════════════════════════════════════
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    if MODEL is None:
        return jsonify({
            'ok': False,
            'error': 'MODEL_NOT_LOADED',
            'message': '서버에 모델이 로드되지 않았습니다.'
        }), 503

    try:
        payload = request.get_json(force=True, silent=True) or {}
        user_name = payload.pop('user_name', None)
        user_age_raw = payload.pop('user_age', None)
        try:
            user_age = int(user_age_raw) if user_age_raw is not None else None
        except (ValueError, TypeError):
            user_age = None

        # 예측
        pred_class, proba, X = predict(payload)
        pred_label = LABEL_NAMES.get(pred_class, '알 수 없음')

        # ★ 구버전 기능 복원: 상위 3개 기여 변수 (explain 기능)
        top_reasons = []
        if EXPLAINER_MODEL is not None:
            try:
                top_reasons = compute_top_reasons(X, pred_class, payload, top_k=3)
            except Exception as exp_err:
                print(f"[EXPLAIN] 실패 (예측은 정상): {exp_err}")
                top_reasons = []

        print(f"[PREDICT] 예측 클래스: {pred_class} ({pred_label})")
        print(f"[PREDICT] 확률: {proba}")
        if top_reasons:
            print(f"[PREDICT] 상위 이유: {[r['feature_kor'] for r in top_reasons]}")

        # 위험 수준 판정
        risk_level = 'low'
        if pred_class == 2:
            risk_level = 'high'
        elif pred_class == 1:
            risk_level = 'medium'
        # 자살 계획 응답이 '예'면 무조건 high
        if payload.get('BP6_2') in (1, '1'):
            risk_level = 'high'

        # DB 저장
        try:
            new_record = Survey(
                user_name=user_name,
                user_age=user_age,
                payload=json.dumps(payload, ensure_ascii=False),
                pred_class=pred_class,
                pred_label=pred_label,
                proba=json.dumps(proba),
            )
            db.session.add(new_record)
            db.session.commit()
        except Exception as db_err:
            print(f"[PREDICT] DB 저장 실패 (예측은 정상): {db_err}")
            db.session.rollback()

        return jsonify({
            'ok': True,
            'pred_class' : pred_class,
            'pred_label' : pred_label,
            'proba'      : proba,
            'risk_level' : risk_level,
            'message'    : make_message(pred_class, risk_level),
            'top_reasons': top_reasons,        # ★ 구버전 기능 복원
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'ok': False,
            'error': type(e).__name__,
            'message': f'예측 중 오류: {str(e)}'
        }), 500


# ════════════════════════════════════════════════════════
# ★ 구버전 기능 복원: /explain 별도 라우트
# ════════════════════════════════════════════════════════
@app.route('/explain', methods=['POST'])
def explain_route():
    if MODEL is None:
        return jsonify({
            'ok': False,
            'error': 'MODEL_NOT_LOADED',
            'message': '서버에 모델이 로드되지 않았습니다.'
        }), 503

    if EXPLAINER_MODEL is None:
        return jsonify({
            'ok': False,
            'error': 'EXPLAINER_NOT_AVAILABLE',
            'message': '설명용 모델이 번들에 없습니다.'
        }), 503

    try:
        payload = request.get_json(force=True, silent=True) or {}
        payload.pop('user_name', None)
        payload.pop('user_age', None)

        pred_class, proba, top_reasons = explain_prediction(payload, top_k=3)
        pred_label = LABEL_NAMES.get(pred_class, '알 수 없음')

        return jsonify({
            'ok': True,
            'pred_class' : pred_class,
            'pred_label' : pred_label,
            'proba'      : proba,
            'top_reasons': top_reasons,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'ok': False,
            'error': type(e).__name__,
            'message': f'설명 중 오류: {str(e)}'
        }), 500


def make_message(pred_class, risk_level):
    if pred_class == 0:
        return ('현재 응답을 바탕으로는 우울 위험 신호가 두드러지지 않습니다. '
                '다만 이 결과는 의학적 진단을 대체하지 않으며, 마음이 힘들 땐 언제든 도움을 받을 수 있다는 점을 기억해주세요.')
    elif pred_class == 1:
        return ('가벼운 우울 신호가 관찰됩니다. 일상 속 작은 변화(규칙적인 수면, 가벼운 운동, 신뢰하는 사람과의 대화)가 도움이 될 수 있어요. '
                '증상이 지속되거나 심해진다면 전문가 상담을 권해드립니다.')
    else:
        return ('우울 신호가 비교적 뚜렷하게 나타나고 있어요. 혼자 견디지 마시고, '
                '전문가 상담이나 가까운 정신건강복지센터의 도움을 받아보시길 권유드립니다.')


@app.route('/chat', methods=['POST'])
def chat_route():
    """
    챗봇 대화 엔드포인트

    Request (JSON):
      {
        "message": "오늘 너무 힘들어요",
        "history": [
          {"role": "user", "text": "..."},
          {"role": "model", "text": "..."}
        ],
        "context": {
          "user_name": "홍길동",
          "pred_label": "경도(5~9)",
          "pred_class": 1,
          "proba": {"0": 0.2, "1": 0.5, "2": 0.3}
        }
      }

    Response (JSON):
      {
        "ok": true,
        "reply": "...",
        "crisis": false
      }
    """
    if GEMINI_CLIENT is None:
        return jsonify({
            'ok': False,
            'error': 'CHATBOT_NOT_AVAILABLE',
            'reply': '챗봇이 현재 사용 불가능합니다. 관리자에게 문의해주세요.'
        }), 503

    try:
        data = request.get_json(force=True, silent=True) or {}
        user_message = (data.get('message') or '').strip()
        history = data.get('history') or []
        context = data.get('context') or {}

        if not user_message:
            return jsonify({'ok': False, 'reply': '메시지가 비어있어요.'}), 400

        # 위기 신호 감지
        crisis_detected = detect_crisis(user_message)

        # 시스템 프롬프트 구성
        system_prompt = build_system_prompt(
            user_name=context.get('user_name'),
            pred_label=context.get('pred_label'),
            pred_class=context.get('pred_class'),
            proba=context.get('proba'),
        )

        # 대화 이력 → Gemini Content 형식으로 변환
        contents = []
        for turn in history:
            role = turn.get('role')
            text = (turn.get('text') or '').strip()
            if not text:
                continue
            gemini_role = 'user' if role == 'user' else 'model'
            contents.append(
                genai_types.Content(
                    role=gemini_role,
                    parts=[genai_types.Part.from_text(text=text)]
                )
            )
        # 현재 사용자 메시지 추가
        contents.append(
            genai_types.Content(
                role='user',
                parts=[genai_types.Part.from_text(text=user_message)]
            )
        )

        # Gemini API 호출
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=400,
            ),
        )

        reply_text = (response.text or '').strip() if hasattr(response, 'text') else ''

        if not reply_text:
            reply_text = '잠시 마음을 정리하고 다시 답해드릴게요.'

        return jsonify({
            'ok': True,
            'reply': reply_text,
            'crisis': crisis_detected,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'ok': False,
            'error': type(e).__name__,
            'reply': '대화 중 문제가 발생했어요. 잠시 후 다시 시도해주세요.',
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'ok': True,
        'model_loaded': MODEL is not None,
        'chatbot_loaded': GEMINI_CLIENT is not None,
        'explainer_loaded': EXPLAINER_MODEL is not None,    # ★ 구버전 기능 복원
        'feature_cols_raw_count': len(FEATURE_COLS_RAW) if FEATURE_COLS_RAW else 0,
        'feature_cols_onehot_count': len(FEATURE_COLS_ONEHOT) if FEATURE_COLS_ONEHOT else 0,
        'time': datetime.now().isoformat(),
    })


# ════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
