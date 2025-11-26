import streamlit as st
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import joblib
import os
import warnings
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# ---------------------------------------------
# 설정 및 상수
# ---------------------------------------------
REPO_ID = "rlawltjd/ventilation_predict"
filenames = {
    "model": "models_catboost.pkl",
    "imputer": "imputer_median.pkl",
    "features": "feature_cols.pkl",
    "threshold": "threshold.pkl"
}

try:
    AIRKOREA_API_KEY = st.secrets["AIRKOREA_API_KEY"]
    KMA_API_KEY = st.secrets["KMA_API_KEY"]
    SEOUL_TRAFFIC_API_KEY = st.secrets["SEOUL_TRAFFIC_API_KEY"]
except:
    st.error("API 키가 설정되지 않았습니다. Streamlit Secrets를 확인해주세요.")

KEY_MAP_AIRKOREA = {
    'SO2': 'SPDX', 'CO': 'CBMX', 'O3': 'OZON',
    'NO2': 'NTDX', 'PM10': 'PM', 'PM25': 'FPM'
}

TARGETS = ["PM25_t_plus_1", "PM25_t_plus_2", "PM25_t_plus_3"]


st.set_page_config(
    page_title="스마트 환기 알리미",
    page_icon="🌬️",
    layout="centered"
)



@st.cache_resource
def load_artifacts_from_hf():
    paths = {}
    try:
        for key, filename in filenames.items():
            file_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
            paths[key] = file_path
        
        models = joblib.load(paths["model"])
        imputer = joblib.load(paths["imputer"])
        feature_cols = joblib.load(paths["features"])
        thresh = joblib.load(paths["threshold"])
        
        return models, imputer, feature_cols, thresh
    except Exception as e:
        st.error(f"모델 다운로드 실패: {e}")
        return None, None, None, None

def safe_float(val):
    try:
        f_val = float(val)
        return np.nan if f_val < -8.0 else f_val
    except (ValueError, TypeError):
        return np.nan

def get_air_data(target_dt):
    tm_str = target_dt.strftime("%Y%m%d%H00")
    url = f"http://openAPI.seoul.go.kr:8088/{AIRKOREA_API_KEY}/json/TimeAverageAirQuality/1/5/{tm_str}/동대문구"
    try:
        response = requests.get(url, timeout=5)
        data_dict = response.json()
        if 'TimeAverageAirQuality' in data_dict and data_dict['TimeAverageAirQuality']['list_total_count'] > 0:
            api_row = data_dict['TimeAverageAirQuality']['row'][0]
            air_data_final = {}
            for model_key, api_key in KEY_MAP_AIRKOREA.items():
                air_data_final[model_key] = safe_float(api_row.get(api_key))
            return air_data_final
        return None
    except:
        return None

def get_weather_data(target_dt):
    tm_str = target_dt.strftime("%Y%m%d%H00")
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm={tm_str}&stn=108&dataType=JSON&authKey={KMA_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        data_lines = [line for line in response.text.splitlines() if not line.startswith('#')]
        if not data_lines or not data_lines[0].strip():
            return None
        values = data_lines[0].split()
        weather_data = {
            "WS": safe_float(values[3]), "PS": safe_float(values[8]),
            "TA": safe_float(values[11]), "HM": safe_float(values[13]),
            "RN": safe_float(values[15]), "VS": safe_float(values[32]),
            "WD_raw": safe_float(values[2]),
        }
        if weather_data.get("RN") == -9.0: weather_data["RN"] = 0.0
        return weather_data
    except:
        return None

def get_traffic_data(target_dt):
    tm_str = target_dt.strftime("%Y%m%d%H00")
    date_str, hour_str = target_dt.strftime("%Y%m%d"), target_dt.strftime("%H")
    url = f"http://openapi.seoul.go.kr:8088/{SEOUL_TRAFFIC_API_KEY}/xml/VolInfo/1/100/F-05/{date_str}/{hour_str}/"
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        rows = root.findall('row')
        if not rows: return None
        total_vol = sum(int(row.find('vol').text) for row in rows if row.find('vol').text)
        return {"Traffic": float(total_vol)}
    except:
        return None

def clip_range(s, low=None, high=None):
    s = pd.to_numeric(s, errors="coerce").copy()
    if low is not None: s[s < low] = low
    if high is not None: s[s > high] = high
    return s

def create_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "HM" in df.columns: df["HM"] = clip_range(df["HM"], 0, 100)
    for col in ["PM25","PM10","SO2","NO2","O3","CO","WS","RN"]:
        if col in df.columns: df[col] = clip_range(df[col], 0, None)
            
    df['WD_sin'], df['WD_cos'] = 0.0, 0.0
    if 'WD_raw' in df.columns:
        mask = df['WD_raw'].notna() & (df['WD_raw'] > 0)
        if mask.any():
            rads = np.deg2rad(df.loc[mask, 'WD_raw'].replace(360, 0) * 10)
            df.loc[mask, 'WD_sin'] = np.sin(rads)
            df.loc[mask, 'WD_cos'] = np.cos(rads)
        df = df.drop(columns=['WD_raw']) 
    
    if 'Traffic' not in df.columns: df['Traffic'] = np.nan

    candidates = [c for c in df.columns if c not in ["timestamp"] + TARGETS]
    for c in candidates:
        if df[c].dtype == "O": df[c] = pd.to_numeric(df[c], errors="coerce")
    
    num_cols = [c for c in candidates if np.issubdtype(df[c].dtype, np.number)]
    
    df = df.set_index("timestamp")
    if not df.index.is_monotonic_increasing: df = df.sort_index()
    
    df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
    for c in num_cols: df[c] = df[c].fillna(df[c].rolling(3, min_periods=1).median())
    if 'RN' in df.columns: df['RN'] = df['RN'].fillna(0)

    df = df.reset_index()
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dow"] > 5).astype(int) 
    df["is_night"] = ((df["hour"] <= 6) | (df["hour"] >= 22)).astype(int)

    base = [c for c in ["PM25","PM10","NO2","O3","SO2","WS","TA","HM"] if c in df.columns]
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in base:
        df[f"{col}_roll3_mean"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_roll3_std"]  = df[col].rolling(3, min_periods=2).std()
        df[f"{col}_roll6_mean"] = df[col].rolling(6, min_periods=1).mean()
        df[f"{col}_roll6_std"]  = df[col].rolling(6, min_periods=2).std()
        for k in [1,2,3,4,5,6]: df[f"{col}_lag{k}"] = df[col].shift(k)

    if set(["WS","WD_sin","WD_cos"]).issubset(df.columns):
        df["WS_sin"] = df["WS"] * df["WD_sin"]
        df["WS_cos"] = df["WS"] * df["WD_cos"]
    if set(["TA","HM"]).issubset(df.columns):
        df["TAxHM"] = df["TA"] * df["HM"]
        
    LEGACY = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25', 'WS', 'PS', 'TA', 'HM', 'RN', 'VS', 'Traffic', 'WD_sin', 'WD_cos']
    for col in LEGACY:
        if col in df.columns:
            df[f"{col}_t_minus_1"] = df[col].shift(1)
            df[f"{col}_t_minus_2"] = df[col].shift(2)
        else:
            df[f"{col}_t_minus_1"] = np.nan
            df[f"{col}_t_minus_2"] = np.nan
    return df.copy()

def get_status(value, thresh_mod):
    if value <= 15: return "좋음 (Good)", "🟢"
    if value <= thresh_mod: return "보통 (Moderate)", "🟡"
    if value <= 75: return "나쁨 (Bad)", "🟠"
    return "매우 나쁨 (Very Bad)", "🔴"

# ---------------------------------------------
# 메인 앱 로직
# ---------------------------------------------

# 1. 사이드바
with st.sidebar:
    st.header("시스템 상태")
    with st.spinner("모델 다운로드 및 로드 중..."):
        MODELS, IMPUTER, FEAT_COLS, THRESH = load_artifacts_from_hf()
    
    if MODELS:
        st.success(f"모델 로드 완료")
    else:
        st.error(" 모델 로드 실패")

# 2. 메인 화면
st.title("🌬️ 동대문구 환기 알리미")
st.markdown("실시간 대기질, 기상, 교통량 데이터를 분석하여 **향후 3시간 내 최적의 환기 시간**을 알려드립니다.")

if st.button("🚀 실시간 분석 시작", type="primary"):
    if not MODELS:
        st.error("모델이 로드되지 않았습니다. 잠시 후 다시 시도해주세요.")
    else:
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # (1) 데이터 수집
        dt_now = datetime.now()
        data_list = []
        
        status_text.text("📡 실시간 데이터 수집 중 (최근 10시간)...")
        for i, h in enumerate(range(9, -1, -1)):
            dt = dt_now - timedelta(hours=h)
            air = get_air_data(dt) or {}
            weather = get_weather_data(dt) or {}
            traffic = get_traffic_data(dt) or {}
            
            merged = {**air, **weather, **traffic}
            merged['timestamp'] = dt.replace(minute=0, second=0, microsecond=0)
            data_list.append(merged)
            progress_bar.progress((i + 1) * 10)
        
        # (2) 전처리 및 예측
        status_text.text("⚙️ 데이터 분석 및 예측 중...")
        df_raw = pd.DataFrame(data_list)
        
        if len(df_raw) < 7:
            st.error(f"데이터 부족으로 분석 실패 (수집된 데이터: {len(df_raw)}개)")
        else:
            try:
                df_feat = create_features(df_raw)
                X_raw = df_feat[FEAT_COLS].iloc[-1:]
                
                if X_raw.empty:
                    st.error("전처리 후 데이터가 없습니다.")
                else:
                    X_imputed = IMPUTER.transform(X_raw)
                    X_final = pd.DataFrame(X_imputed, columns=FEAT_COLS, index=X_raw.index)
                    
                    preds = {}
                    for k in ['PM25_t_plus_1', 'PM25_t_plus_2', 'PM25_t_plus_3']:
                        preds[k] = MODELS[k].predict(X_final)[0]
                    
                    status_text.text("✅ 분석 완료!")
                    progress_bar.progress(100)
                    
                    st.divider()
                    
                    col1, col2, col3 = st.columns(3)
                    times = [(dt_now + timedelta(hours=i)).strftime("%H시") for i in [1, 2, 3]]
                    p_vals = [preds['PM25_t_plus_1'], preds['PM25_t_plus_2'], preds['PM25_t_plus_3']]
                    
                    infos = []
                    for i, (t_str, val) in enumerate(zip(times, p_vals)):
                        txt, icon = get_status(val, THRESH)
                        infos.append({'time': t_str, 'val': val, 'txt': txt})
                        with [col1, col2, col3][i]:
                            st.metric(label=f"{t_str} 예측", value=f"{val:.1f} µg/m³")
                            st.caption(f"{icon} {txt}")
                    
                    st.subheader("📢 분석 결과")
                    
                    good_times = [x for x in infos if x['val'] <= THRESH]
                    if good_times:
                        best = min(good_times, key=lambda x: x['val'])
                        st.success(f"**환기 추천!** {best['time']}가 가장 좋습니다.\n\n(예측 농도: {best['val']:.1f} µg/m³)")
                    else:
                        st.warning("**환기 자제 권고**\n\n향후 3시간 동안 미세먼지 농도가 '나쁨' 수준일 것으로 예측됩니다.")
                    
                    with st.expander("상세 수집 데이터 보기"):
                        st.dataframe(df_raw.tail(7))

            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")