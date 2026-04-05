import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# 기존에 작성한 단일 파일을 라이브러리처럼 임포트합니다.
import gas_common_model_v3 as gcm

# 웹페이지 기본 설정
st.set_page_config(page_title="가스 시설물 실시간 대시보드", layout="wide")


@st.cache_resource
def get_artifacts():
    base_dir = Path(gcm.__file__).resolve().parent
    save_dir = base_dir / 'common_model_artifacts'
    return gcm.load_artifacts(str(save_dir))


@st.cache_data
def get_data():
    base_dir = Path(gcm.__file__).resolve().parent
    excel_path = base_dir / '01. 시설물 50개 샘플 데이터.xlsx'
    return gcm.build_master_dataset(str(excel_path))


def main():
    st.title("📊 가스 시설물 실시간 모니터링 대시보드")

    # 1. 아티팩트 및 데이터 로드
    try:
        model, scaler_map, thresholds, config = get_artifacts()
        master_df = get_data()
    except Exception as e:
        st.error(f"데이터나 모델을 불러오지 못했습니다. 원본 파이썬 파일을 한 번 실행하여 모델을 학습시켜주세요.\n오류: {e}")
        return

    # 2. 사이드바 - 기기 선택 UI
    device_list = sorted(master_df['장비번호'].unique())
    st.sidebar.header("모니터링 설정")
    selected_device = st.sidebar.selectbox("모니터링할 장비 번호 선택", device_list)

    # 선택된 기기 데이터 필터링
    device_df = master_df[master_df['장비번호'] == selected_device].sort_values('측정시각').copy()
    if device_df.empty:
        st.warning("선택한 기기의 데이터가 없습니다.")
        return

    # 3. 최신 상태 요약 표시
    st.subheader(f"📍 [{selected_device}] 현재 센서 측정값 (최신)")
    last_record = device_df.iloc[-1]

    cols = st.columns(len(gcm.BASE_FEATURES))
    for i, feat in enumerate(gcm.BASE_FEATURES):
        val = last_record[feat]
        cols[i].metric(label=feat, value=f"{val:.2f}" if pd.notnull(val) else "N/A")

    time_steps = config['time_steps']

    # ==========================================
    # 4. [신규 추가] 시계열 이상 탐지 결과 그래프
    # ==========================================
    st.subheader("🚨 이상 탐지(Anomaly Detection) 결과 추이")

    if len(device_df) > time_steps:
        # 모델 예측을 위한 전체 데이터 전처리
        ddf = gcm.assign_normalization_group(device_df)
        ddf = gcm.add_engineered_features(ddf, gcm.BASE_FEATURES)

        group_name = str(ddf['정규화그룹'].iloc[0])

        if group_name in scaler_map:
            scaler = scaler_map[group_name]
            scaled_data = scaler.transform(ddf[config['feature_columns']].astype(float).values)

            # 시퀀스 생성 및 모델 예측
            X = gcm.create_sequences(scaled_data, time_steps)
            preds = model.predict(X, verbose=0)

            # 각 시점별 오차(MSE) 계산
            mses = np.mean(np.power(X - preds, 2), axis=(1, 2))

            # 그래프용 날짜 데이터 (앞부분 time_steps 만큼 잘려나감)
            plot_dates = ddf['측정시각'].iloc[time_steps:].values

            # 해당 기기의 임계치(Threshold)
            device_threshold = float(thresholds.get(selected_device, np.mean(list(thresholds.values()))))

            # Plotly로 시각화
            fig_anomaly = go.Figure()

            # 오차(MSE) 꺾은선 그래프
            fig_anomaly.add_trace(go.Scatter(
                x=plot_dates, y=mses,
                mode='lines',
                name='오차 점수(MSE)',
                line=dict(color='blue', width=2)
            ))

            # 한계치(Threshold) 기준선 (빨간 점선)
            fig_anomaly.add_trace(go.Scatter(
                x=[plot_dates[0], plot_dates[-1]], y=[device_threshold, device_threshold],
                mode='lines',
                name='위험 한계치(Threshold)',
                line=dict(color='red', width=2, dash='dash')
            ))

            fig_anomaly.update_layout(
                title=f"모델 복원 오차(MSE) 변화량 - 임계치: {device_threshold:.6f}",
                xaxis_title="측정 시각",
                yaxis_title="MSE (오차율)",
                hovermode="x unified",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig_anomaly, use_container_width=True)

            # 최근 데이터 기준 텍스트 알림
            current_mse = mses[-1]
            if current_mse > device_threshold:
                st.error(f"⚠️ **경고:** 가장 최근 데이터에서 이상 징후가 감지되었습니다! (현재 오차: {current_mse:.6f})")
            else:
                st.success(f"✅ **정상:** 현재 위험 한계치 아래에서 안정적으로 작동 중입니다. (현재 오차: {current_mse:.6f})")
        else:
            st.warning("이 기기가 속한 그룹의 스케일러(Scaler) 정보가 없습니다.")
    else:
        st.info(f"데이터 부족 (최소 {time_steps}개의 시계열 포인트가 필요합니다.)")

    # ==========================================
    # 5. 피처별 인터랙티브 시계열 그래프
    # ==========================================
    st.subheader("📈 원본 센서 피처별 변화 추이")

    # 기본적으로 온도, 습도 2개만 먼저 선택되어 있도록 설정 (원하는 대로 다중 선택 가능)
    selected_feats = st.multiselect("확인할 피처 선택", gcm.BASE_FEATURES, default=['온도', '습도'])

    if selected_feats:
        # 선택된 피처 개수만큼 반복하며 각각의 그래프를 아래로 나열하여 그립니다.
        for feat in selected_feats:
            fig = px.line(device_df, x='측정시각', y=feat, title=f"[{feat}] 변화 추이")

            # 그래프 디자인 설정 (높이를 살짝 줄여서 여러 개 보기 편하게 만듦)
            fig.update_layout(
                hovermode="x unified",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title=feat,
                xaxis_title=""  # x축 이름(측정시각)은 굳이 안 써도 되니 깔끔하게 생략
            )

            # 각 피처의 선 색상을 다르게 하고 싶다면 아래 주석을 해제하고 적용할 수 있습니다.
            # fig.update_traces(line_color='green')

            st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()