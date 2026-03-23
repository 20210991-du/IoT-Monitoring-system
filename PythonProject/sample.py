import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta


def generate_sample_csv(file_name='sensor_logs_sample.csv', num_rows=1500):  # 1500개로 증가
    start_time = datetime(2026, 3, 23, 2, 0, 0)
    data = []

    for i in range(num_rows):
        timestamp = start_time + timedelta(minutes=10 * i)

        # 144 스텝 = 24시간 (10분 간격). 하루 주기의 완만한 사인파 생성
        time_factor = i * (2 * math.pi / 144)

        # 큰 흐름(주기성) + 아주 미세한 노이즈(0.5 수준)
        curr_potential = -1000.0 + 50 * math.sin(time_factor) + np.random.uniform(-0.5, 0.5)
        curr_temp = 13.1 + 5 * math.cos(time_factor) + np.random.uniform(-0.1, 0.1)
        curr_battery = 3600.0 - (i * 0.05)  # 배터리는 일정하게 감소

        row = {
            '시설명': 'A구역 가스기지',
            '장비명': 'RTU-01',
            '설치위치': '지하 매설구간',
            '측정상태': '정상',
            '방식전위': f"{curr_potential:,.1f}mV",  # 소수점 첫째자리까지 살려서 더 정밀하게
            '방식전류': "2mA",  # 전류, AC유입 등은 고정 또는 최소 변동으로 단순화
            'AC유입': f"{320 + np.random.uniform(-1, 1):.1f}mV",
            '배터리': f"{curr_battery:,.1f}mV",
            '온도': f"{curr_temp:.1f}℃",
            '습도': f"{5.5 + np.random.uniform(-0.1, 0.1):.1f}%",
            '통신상태': '정상',
            '최근 갱신일시': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
        data.append(row)

    df = pd.DataFrame(data)

    # 이상치 및 결측치 삽입 유지
    df.loc[1200:1205, '방식전위'] = "-500.0mV"
    df.loc[5:7, '방식전위'] = np.nan

    df.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"{file_name} 생성 완료 (정밀 패턴 버전)!")


generate_sample_csv()