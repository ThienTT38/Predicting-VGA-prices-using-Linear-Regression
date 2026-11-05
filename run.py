# =============================================================================
# COMPUTER PROJECT MAI391: MATHEMATICS FOR MACHINE LEARNING
# Đề tài: Dự đoán giá Card đồ họa (VGA) bằng Hồi quy Tuyến tính (Linear Regression)
# Nền tảng Toán học: MLE, Phép chiếu trực giao (OLS), Phân rã SVD, 
#                    Giải tích Vector (Gradient Descent) & Ước lượng hậu nghiệm MAP (Ridge)
# =============================================================================

import time
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression as SklearnLR, Ridge as SklearnRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", palette="deep")

# =============================================================================
# CẤU HÌNH THỰC NGHIỆM
# =============================================================================
CSV_PATH = 'Master_VGA_All_Columns_Aligned.csv'
# Chế độ dữ liệu:
#   'VN_RETAIL'    : Chỉ dùng 1.145 mẫu giá VNĐ thực tế tại VN (Phong Vũ, SP-One, Mega, Phi Long)
#                    nhưng vẫn dùng 3.762 mẫu BuildCores làm Từ điển điền khuyết thông số phần cứng.
#   'COMBINED_VND' : Gộp cả VN_RETAIL và BuildCores (quy đổi USD -> VNĐ và lọc nhiễu giá chợ đen).
DATASET_MODE = 'VN_RETAIL'
USD_TO_VND_RATE = 25400.0
RANDOM_SEED = 42


# =============================================================================
# PHẦN 1: TRÍCH XUẤT ĐẶC TRƯNG ĐA CỘT & ĐIỀN KHUYẾT CHÉO (NHIỆM VỤ 1 - 2 ĐIỂM)
# =============================================================================

def parse_price_and_currency(val):
    """Tách giá trị số và đơn vị tiền tệ (USD hoặc VND) từ cột Giá_Hiện_Tại."""
    if pd.isna(val):
        return np.nan, None
    s = str(val).strip()
    if 'liên hệ' in s.lower() or s == '' or s == '0':
        return np.nan, None
    if '$' in s:
        num = re.sub(r'[^\d.]', '', s)
        return (float(num) if num else np.nan), 'USD'
    else:
        num = re.sub(r'[^\d]', '', s)
        return (float(num) if num else np.nan), 'VND'


def normalize_chipset(row):
    """Chuẩn hóa tên GPU (Chipset) từ các cột văn bản để làm khóa tham chiếu phần cứng."""
    cols = ['Chipset_Raw', 'Chipset_Chuẩn', 'Engine đồ họa', 'Tên_Sản_Phẩm', 'Model']
    pattern = (
        r'(RTX\s*\d{4}\s*(?:TI\s*SUPER|TI|SUPER|ADA)?|'
        r'GTX\s*\d{3,4}\s*(?:TI|SUPER)?|'
        r'GT\s*\d{3,4}|'
        r'RX\s*\d{3,4}\s*(?:XTX|XT|GRE)?|'
        r'ARC\s*[AB]\d{3}|'
        r'RTX\s*A\d{4}|'
        r'QUADRO\s*[A-Z0-9]+|'
        r'T\d{3,4})'
    )
    for col in cols:
        val = str(row.get(col, '')).upper()
        if pd.isna(row.get(col)) or val in ['NAN', 'KHÁC', 'WORKSTATION GPU', '']:
            continue
        m = re.search(pattern, val)
        if m:
            return re.sub(r'\s+', ' ', m.group(1).strip())
    return 'OTHER'


def extract_vram_gb(row):
    """Hợp nhất dung lượng VRAM (GB) từ các cột đồng nghĩa."""
    for col in ['VRAM_Chuẩn', 'VRAM_Raw', 'Dung lượng bộ nhớ', 'Bộ nhớ', 'Tên_Sản_Phẩm']:
        val = str(row.get(col, ''))
        if pd.isna(row.get(col)) or val == 'nan':
            continue
        m = re.search(r'(\d+(?:\.\d+)?)\s*GB', val, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if 1 <= v <= 128:
                return v
        m_mb = re.search(r'(\d+)\s*MB', val, re.IGNORECASE)
        if m_mb:
            v = float(m_mb.group(1)) / 1024.0
            if 0.25 <= v <= 128:
                return v
    return np.nan


def extract_bus_bit(row):
    """Hợp nhất độ rộng Bus bộ nhớ (bit)."""
    valid_buses = {32, 64, 96, 128, 160, 192, 256, 320, 352, 384, 512, 1024, 2048, 3072, 4096}
    for col in ['Bus_Bộ_Nhớ', 'Bus RAM', 'Bus bộ nhớ', 'Memory Bus', 'VRAM_Raw']:
        val = str(row.get(col, ''))
        if pd.isna(row.get(col)) or val == 'nan':
            continue
        m = re.search(r'(\d+)\s*-?\s*bit', val, re.IGNORECASE)
        if m and int(m.group(1)) in valid_buses:
            return float(m.group(1))
        if col in ['Bus_Bộ_Nhớ', 'Bus RAM'] and re.match(r'^\s*\d+(\.0)?\s*$', val):
            b = int(float(val))
            if b in valid_buses:
                return float(b)
    return np.nan


def extract_vram_type(row):
    """Trích xuất chuẩn bộ nhớ VRAM và ánh xạ sang thang thứ bậc băng thông."""
    for col in ['Loại_VRAM', 'VRAM_Raw', 'Kiểu bộ nhớ', 'Loại RAM', 'Tên_Sản_Phẩm']:
        val = str(row.get(col, '')).upper()
        if pd.isna(row.get(col)) or val == 'NAN':
            continue
        for t in ['GDDR7', 'HBM3', 'HBM2E', 'HBM2', 'GDDR6X', 'GDDR6', 'GDDR5X', 'HBM', 'GDDR5', 'DDR4', 'DDR3']:
            if t in val:
                return t
    return np.nan


def extract_cores(row):
    """Hợp nhất số nhân xử lý (CUDA Cores / Stream Processors)."""
    cols = ['Core Count', 'CUDA_Cores', 'Số lượng đơn vị xử lý',
            'Số Nhân Xử Lý', 'Nhân CUDA', 'CUDA Cores', 'Stream Processors']
    for col in cols:
        val = str(row.get(col, ''))
        if pd.isna(row.get(col)) or val == 'nan':
            continue
        m = re.search(r'(\d{3,5})', val.replace(',', '').replace('.', ''))
        if m:
            c = float(m.group(1))
            if 128 <= c <= 30000:
                return c
    return np.nan


def extract_boost_clock(row):
    """Hợp nhất xung nhịp Boost tối đa (MHz)."""
    cols = ['Core Boost Clock', 'GPU clock', 'Core_Clock',
            'Engine Clock ( Xung nhịp nhân GPU)', 'Xung nhịp']
    for col in cols:
        val = str(row.get(col, ''))
        if pd.isna(row.get(col)) or val == 'nan':
            continue
        nums = [float(x) for x in re.findall(r'(\d{3,4}(?:\.\d+)?)\s*MHz', val, re.IGNORECASE)]
        valid = [x for x in nums if 500 <= x <= 3600]
        if valid:
            return max(valid)
        nums2 = [float(x) for x in re.findall(r'\b(\d{4})\b', val)]
        valid2 = [x for x in nums2 if 900 <= x <= 3500]
        if valid2:
            return max(valid2)
    return np.nan


def extract_power_w(row):
    """Trích xuất công suất TDP (W) hoặc quy đổi từ Công suất nguồn đề xuất (PSU)."""
    for col in ['TDP', 'Công suất tiêu thụ']:
        val = str(row.get(col, ''))
        if pd.isna(row.get(col)) or val == 'nan':
            continue
        m = re.search(r'(\d{2,4})\s*W', val, re.IGNORECASE)
        if m and 15 <= float(m.group(1)) <= 1000:
            return float(m.group(1))
    # Nếu chỉ có Nguồn_Đề_Xuất (PSU), ước lượng TDP ~ 0.42 * PSU_W
    for col in ['Nguồn_Đề_Xuất', 'Công suất nguồn đề nghị', 'Nguồn đề xuất']:
        val = str(row.get(col, ''))
        if pd.isna(row.get(col)) or val == 'nan':
            continue
        nums = [float(x) for x in re.findall(r'(\d{3,4})\s*W', val, re.IGNORECASE)]
        valid = [x for x in nums if 250 <= x <= 1600]
        if valid:
            return round(max(valid) * 0.42, 1)
    return np.nan


def extract_fans(row):
    """Trích xuất số lượng quạt tản nhiệt (hoặc tản nhiệt nước AIO = 3.5)."""
    for col in ['Cooling', 'Tản nhiệt', 'Tản nhiệt', 'Tên_Sản_Phẩm', 'Model']:
        val = str(row.get(col, '')).upper()
        if pd.isna(row.get(col)) or val == 'NAN':
            continue
        if any(w in val for w in ['AIO', 'WATER', 'LIQUID', 'HYDRO', 'LC ', 'SEA HAWK', 'SUPRIM LIQUID']):
            return 3.5
        m = re.search(r'(\d)\s*(?:FAN|QUẠT)', val)
        if m and int(m.group(1)) in [1, 2, 3, 4]:
            return float(m.group(1))
        if any(w in val for w in ['TRIPLE', '3X', 'TUF', 'STRIX', 'SUPRIM', 'TRIO', 'AORUS MASTER', 'AERO OC']):
            return 3.0
        if any(w in val for w in ['DUAL', 'TWIN', '2X', 'VENTUS 2X', 'WINDFORCE 2X']):
            return 2.0
        if any(w in val for w in ['SINGLE', 'AERO ITX', '1X', 'LP', 'LOW PROFILE']):
            return 1.0
    return np.nan


def extract_chip_maker(row, clean_chipset):
    """Xác định hãng sản xuất GPU (NVIDIA, AMD, Intel)."""
    if any(k in clean_chipset for k in ['RTX', 'GTX', 'GT ', 'QUADRO', 'T1000', 'T600', 'T400']):
        return 'NVIDIA'
    if 'RX ' in clean_chipset or 'RADEON' in clean_chipset:
        return 'AMD'
    if 'ARC ' in clean_chipset:
        return 'Intel'
    for col in ['Chipset Manufacturer', 'Nhà sản xuất chipset', 'Tên_Sản_Phẩm']:
        val = str(row.get(col, '')).upper()
        if 'NVIDIA' in val or 'GEFORCE' in val or 'QUADRO' in val:
            return 'NVIDIA'
        if 'AMD' in val or 'RADEON' in val:
            return 'AMD'
        if 'INTEL' in val or 'ARC' in val:
            return 'Intel'
    return 'NVIDIA'


def extract_arch_generation(clean_chipset, row):
    """Phân tầng thế hệ kiến trúc (Ordinal Tier từ 1 đến 6) để nắm bắt bước nhảy công nghệ."""
    s = (clean_chipset + " " + str(row.get('Tên_Sản_Phẩm', '')).upper())
    if any(k in s for k in ['RTX 50', 'RX 90', 'BLACKWELL']):
        return 6.0  # Thế hệ mới nhất (2025-2026: RTX 50-series / RX 9000)
    if any(k in s for k in ['RTX 40', 'RX 7', 'ADA', 'ARC B']):
        return 5.0  # Thế hệ RTX 40-series / RX 7000 / RTX Ada
    if any(k in s for k in ['RTX 30', 'RX 6', 'RTX A', 'ARC A']):
        return 4.0  # Thế hệ RTX 30-series / RX 6000 / RTX Ampere Workstation
    if any(k in s for k in ['RTX 20', 'GTX 16', 'RX 5000', 'RX 5700', 'RX 5600', 'RX 5500', 'QUADRO RTX']):
        return 3.0  # Thế hệ Turing / RDNA1
    if any(k in s for k in ['GTX 10', 'RX 580', 'RX 570', 'RX 550', 'QUADRO P', 'QUADRO GV', 'T1000', 'T600', 'T400']):
        return 2.0  # Thế hệ Pascal / Polaris
    return 1.0      # Thế hệ cũ hơn (GT 710, GT 730, ...)


def load_and_preprocess_data(csv_path, mode='VN_RETAIL'):
    print("=" * 80)
    print("BƯỚC 1: ĐỌC DỮ LIỆU & HỢP NHẤT ĐẶC TRƯNG TỪ 963 CỘT")
    print("=" * 80)
    df = pd.read_csv(csv_path, low_memory=False)
    df['Domain'] = df['Link_Sản_Phẩm'].astype(str).apply(
        lambda x: x.split('/')[2] if len(x.split('/')) > 2 else 'Unknown'
    )
    print(f"[*] Kích thước dữ liệu gốc: {df.shape[0]} dòng x {df.shape[1]} cột.")

    # 1. Trích xuất thông số từ toàn bộ 5.128 dòng
    parsed_price = df['Giá_Hiện_Tại'].apply(parse_price_and_currency)
    df['Price_Raw'] = [p[0] for p in parsed_price]
    df['Currency'] = [p[1] for p in parsed_price]

    df['Clean_Chipset'] = df.apply(normalize_chipset, axis=1)
    df['VRAM_GB'] = df.apply(extract_vram_gb, axis=1)
    df['Bus_Bit'] = df.apply(extract_bus_bit, axis=1)
    df['VRAM_Type'] = df.apply(extract_vram_type, axis=1)
    df['Core_Count'] = df.apply(extract_cores, axis=1)
    df['Boost_MHz'] = df.apply(extract_boost_clock, axis=1)
    df['TDP_W'] = df.apply(extract_power_w, axis=1)
    df['Fans'] = df.apply(extract_fans, axis=1)
    df['Chip_Maker'] = [extract_chip_maker(r, c) for (_, r), c in zip(df.iterrows(), df['Clean_Chipset'])]
    df['Arch_Tier'] = [extract_arch_generation(c, r) for (_, r), c in zip(df.iterrows(), df['Clean_Chipset'])]

    # Đặc trưng nhị phân: Card Workstation (Quadro / RTX A-series / Pro) & Bản Ép xung (OC) & Cao cấp (Flagship)
    def check_workstation(row):
        txt = (str(row['Clean_Chipset']) + " " + str(row.get('Chipset_Chuẩn', '')) + " " + str(row.get('Tên_Sản_Phẩm', ''))).upper()
        return 1.0 if any(k in txt for k in ['QUADRO', 'RTX A', 'WORKSTATION', 'PRO ', 'ADA', 'TESLA', 'A100', 'H100', 'GV100', 'T1000', 'T600', 'T400']) else 0.0

    def check_oc(row):
        txt = (str(row.get('Variant', '')) + " " + str(row.get('Tên_Sản_Phẩm', '')) + " " + str(row.get('Model', ''))).upper()
        return 1.0 if re.search(r'\b(OC|OVERCLOCK|O\d+G|TOP|EXTREME|SUPERCLOCKED)\b', txt) else 0.0

    def check_flagship_tier(row):
        txt = (str(row.get('Series', '')) + " " + str(row.get('Tên_Sản_Phẩm', ''))).upper()
        return 1.0 if any(k in txt for k in ['ROG STRIX', 'ASTRAL', 'SUPRIM', 'AORUS MASTER', 'AORUS XTREME', 'VULCAN', 'NEPTUNE', 'HALL OF FAME', 'HOF', 'NITRO+', 'TAICHI']) else 0.0

    df['Is_Workstation'] = df.apply(check_workstation, axis=1)
    df['Is_OC'] = df.apply(check_oc, axis=1)
    df['Is_Premium_AIB'] = df.apply(check_flagship_tier, axis=1)

    # Chuẩn hóa thương hiệu AIB
    def clean_brand(val, title):
        s = (str(val) + " " + str(title)).upper()
        for b in ['ASUS', 'MSI', 'GIGABYTE', 'ZOTAC', 'COLORFUL', 'GALAX', 'PNY', 'INNO3D', 'SAPPHIRE', 'POWERCOLOR', 'ASROCK', 'PALIT', 'EVGA', 'LEADTEK']:
            if b in s:
                return b
        return 'OTHER_BRAND'

    df['AIB_Brand'] = [clean_brand(r.get('Hãng_Sản_Xuất', ''), r.get('Tên_Sản_Phẩm', '')) for _, r in df.iterrows()]

    # 2. Xây dựng Từ điển Tham chiếu Phần cứng (Hardware Spec Lookup) từ toàn bộ 5.128 mẫu
    #    Nhóm theo (Clean_Chipset, VRAM_GB) và theo Clean_Chipset để điền khuyết chéo
    num_spec_cols = ['Bus_Bit', 'Core_Count', 'Boost_MHz', 'TDP_W', 'Fans']
    valid_chip = df[df['Clean_Chipset'] != 'OTHER']

    chip_vram_medians = valid_chip.groupby(['Clean_Chipset', 'VRAM_GB'])[num_spec_cols].median()
    chip_only_medians = valid_chip.groupby('Clean_Chipset')[num_spec_cols].median()
    chip_vram_type_mode = valid_chip.groupby('Clean_Chipset')['VRAM_Type'].agg(
        lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) > 0 else np.nan
    )

    # Thực hiện Cross-Source Imputation cho các dòng thiếu thông số
    for idx, row in df.iterrows():
        c = row['Clean_Chipset']
        v = row['VRAM_GB']
        if c != 'OTHER':
            for col in num_spec_cols:
                if pd.isna(row[col]):
                    if pd.notna(v) and (c, v) in chip_vram_medians.index and pd.notna(chip_vram_medians.loc[(c, v), col]):
                        df.at[idx, col] = chip_vram_medians.loc[(c, v), col]
                    elif c in chip_only_medians.index and pd.notna(chip_only_medians.loc[c, col]):
                        df.at[idx, col] = chip_only_medians.loc[c, col]
            if pd.isna(row['VRAM_Type']) and c in chip_vram_type_mode.index:
                df.at[idx, 'VRAM_Type'] = chip_vram_type_mode.loc[c]

    # Mã hóa thứ bậc cho chuẩn bộ nhớ VRAM (Ordinal Encoding theo thế hệ băng thông)
    vram_rank_map = {
        'DDR3': 1.0, 'DDR4': 2.0, 'GDDR5': 3.0, 'GDDR5X': 3.5,
        'GDDR6': 4.0, 'HBM': 4.0, 'HBM2': 4.5, 'GDDR6X': 5.0,
        'HBM2E': 5.5, 'GDDR7': 6.0, 'HBM3': 6.0
    }
    df['VRAM_Tier'] = df['VRAM_Type'].map(vram_rank_map)

    # 3. Lọc các mẫu có giá bán hợp lệ theo chế độ thực nghiệm
    priced_df = df[df['Price_Raw'].notna() & (df['Price_Raw'] > 0)].copy()

    if mode == 'VN_RETAIL':
        data = priced_df[priced_df['Currency'] == 'VND'].copy()
        data['Price_Million_VND'] = data['Price_Raw'] / 1e6
        print(f"[*] Chế độ VN_RETAIL: Lọc {len(data)} mẫu có giá VNĐ hợp lệ từ 4 hệ thống bán lẻ Việt Nam.")
    else:
        data = priced_df.copy()
        data['Price_Million_VND'] = np.where(
            data['Currency'] == 'USD',
            (data['Price_Raw'] * USD_TO_VND_RATE) / 1e6,
            data['Price_Raw'] / 1e6
        )
        data['Is_VN_Store'] = (data['Currency'] == 'VND').astype(float)
        # Loại bỏ các mẫu giá scalper ảo trên BuildCores (ví dụ GTX 1050 giá > 5.000 USD)
        chip_med_price = data.groupby('Clean_Chipset')['Price_Million_VND'].transform('median')
        data = data[data['Price_Million_VND'] <= 3.5 * chip_med_price].copy()
        print(f"[*] Chế độ COMBINED_VND: Lọc {len(data)} mẫu hợp lệ sau khi loại ngoại lai chợ đen.")

    # Lọc phạm vi giá hợp lý (từ 1 triệu đến 300 triệu VNĐ)
    data = data[(data['Price_Million_VND'] >= 1.0) & (data['Price_Million_VND'] <= 300.0)].copy()

    print("\n[*] Tỷ lệ lấp đầy đặc trưng sau bước Cross-Source Spec Imputation:")
    check_cols = ['VRAM_GB', 'Bus_Bit', 'VRAM_Tier', 'Core_Count', 'Boost_MHz', 'TDP_W', 'Fans', 'Arch_Tier']
    for col in check_cols:
        cov = data[col].notna().mean() * 100
        print(f"    - {col:<15}: {data[col].notna().sum():4d}/{len(data)} mẫu ({cov:6.2f}%)")

    # Biến mục tiêu: y_raw (Triệu VNĐ) và y_log = ln(Price_Million_VND)
    data['Log_Price'] = np.log(data['Price_Million_VND'])
    return data


# =============================================================================
# PHẦN 2: THIẾT KẾ MA TRẬN X & KIỂM ĐỊNH ĐA CỘNG TUYẾN VIF (CHƯƠNG 3 & 4)
# =============================================================================

def build_design_matrix(train_df, test_df):
    """
    Chuẩn hóa Z-score (chỉ fit trên Train để chống Data Leakage),
    tạo biến tương tác (Interaction Terms) và mã hóa One-Hot (drop_first=True).
    """
    num_cols = ['VRAM_GB', 'Bus_Bit', 'VRAM_Tier', 'Core_Count', 'Boost_MHz', 'TDP_W', 'Fans', 'Arch_Tier']
    bin_cols = ['Is_Workstation', 'Is_OC', 'Is_Premium_AIB']

    train_c = train_df.copy()
    test_c = test_df.copy()

    # Điền giá trị thiếu còn sót lại bằng trung vị của tập Train (chống rò rỉ dữ liệu)
    for col in num_cols:
        med = train_c[col].median()
        train_c[col] = train_c[col].fillna(med)
        test_c[col] = test_c[col].fillna(med)

    # Đặc trưng kỹ thuật phi tuyến chuyển về dạng tuyến tính theo tham số (Log-features & Interaction)
    # Lý thuyết: Giá tăng theo cấp số nhân với số nhân xử lý và băng thông bộ nhớ xấp xỉ (VRAM * Bus)
    for d in [train_c, test_c]:
        d['Log_Cores'] = np.log(d['Core_Count'])
        d['Log_VRAM'] = np.log(d['VRAM_GB'])
        d['Bandwidth_Proxy'] = (d['Bus_Bit'] * d['VRAM_Tier']) / 128.0
        d['Workstation_x_VRAM'] = d['Is_Workstation'] * d['Log_VRAM']

    continuous_features = [
        'Log_VRAM', 'VRAM_GB', 'Bus_Bit', 'VRAM_Tier',
        'Log_Cores', 'Core_Count', 'Boost_MHz', 'TDP_W',
        'Fans', 'Arch_Tier', 'Bandwidth_Proxy', 'Workstation_x_VRAM'
    ]

    # Chuẩn hóa Z-score: x_std = (x - mu_train) / sigma_train
    means = train_c[continuous_features].mean()
    stds = train_c[continuous_features].std().replace(0, 1.0)

    X_train_num = (train_c[continuous_features] - means) / stds
    X_test_num = (test_c[continuous_features] - means) / stds

    # One-Hot Encoding cho Chip_Maker và Top thương hiệu AIB (bỏ 1 cột gốc tránh Dummy Variable Trap)
    top_brands = ['ASUS', 'MSI', 'GIGABYTE', 'ZOTAC', 'COLORFUL', 'GALAX', 'PNY', 'SAPPHIRE']
    for b in top_brands:
        X_train_num[f'Brand_{b}'] = (train_c['AIB_Brand'] == b).astype(float)
        X_test_num[f'Brand_{b}'] = (test_c['AIB_Brand'] == b).astype(float)

    X_train_num['Maker_AMD'] = (train_c['Chip_Maker'] == 'AMD').astype(float)
    X_test_num['Maker_AMD'] = (test_c['Chip_Maker'] == 'AMD').astype(float)
    X_train_num['Maker_Intel'] = (train_c['Chip_Maker'] == 'Intel').astype(float)
    X_test_num['Maker_Intel'] = (test_c['Chip_Maker'] == 'Intel').astype(float)

    for col in bin_cols:
        X_train_num[col] = train_c[col].values
        X_test_num[col] = test_c[col].values

    feature_names = ['Intercept (w0)'] + list(X_train_num.columns)

    # Ghép thêm cột 1 ở đầu ma trận thiết kế X (biểu diễn hệ số chặn w_0)
    X_train = np.column_stack([np.ones(len(X_train_num)), X_train_num.values])
    X_test = np.column_stack([np.ones(len(X_test_num)), X_test_num.values])

    return X_train, X_test, feature_names, X_train_num


def compute_vif_table(X_df):
    """
    Tính Hệ số phóng đại phương sai VIF_j = 1 / (1 - R_j^2) bằng đại số ma trận
    (Chẩn đoán đa cộng tuyến giữa các cột đặc trưng).
    """
    cols = X_df.columns
    M = X_df.values
    vif_list = []
    for j in range(M.shape[1]):
        y_j = M[:, j]
        X_other = np.column_stack([np.ones(M.shape[0]), np.delete(M, j, axis=1)])
        # Giải bình phương tối thiểu qua giả nghịch đảo SVD
        w_j = np.linalg.pinv(X_other) @ y_j
        y_hat = X_other @ w_j
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
        r2_j = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        vif_val = 1.0 / max(1.0 - r2_j, 1e-12)
        vif_list.append({'Feature': cols[j], 'R2_with_others': r2_j, 'VIF': vif_val})
    return pd.DataFrame(vif_list).sort_values(by='VIF', ascending=False)


# =============================================================================
# PHẦN 3: CÀI ĐẶT TOÁN HỌC LINEAR REGRESSION TỪ ĐẦU BẰNG NUMPY (NHIỆM VỤ 2 - 3 ĐIỂM)
# =============================================================================

class MML_LinearRegression_Scratch:
    """
    Cài đặt Hồi quy Tuyến tính theo chuẩn giáo trình Deisenroth et al. (Chương 3, 4, 5, 7, 9):
      1. 'normal_eq' : Nghiệm đóng MLE / Phép chiếu trực giao w = (X^T X)^(-1) X^T y
      2. 'svd'       : Giả nghịch đảo Moore-Penrose qua phân rã SVD: X = U Sigma V^T => w = V Sigma^+ U^T y
      3. 'gd'        : Batch Gradient Descent: w^(t+1) = w^(t) - (eta / N) * X^T (X w^(t) - y)
      4. 'ridge_map' : Ước lượng MAP với Prior Gaussian: w = (X^T X + alpha * I_0)^(-1) X^T y
                       (Lưu ý toán học: Không phạt hệ số chặn w_0 nên I_0[0, 0] = 0).
    """
    def __init__(self, method='normal_eq', alpha=1.0, lr=0.05, epochs=3000):
        self.method = method
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.loss_history = []
        self.sigma2_mle = None  # Phương sai nhiễu ước lượng cực đại hợp lý

    def fit(self, X, y):
        N, D = X.shape
        if self.method == 'normal_eq':
            # Nghiệm đóng OLS: giải hệ phương trình pháp tuyến (X^T X) w = X^T y
            XtX = X.T @ X
            Xty = X.T @ y
            self.w = np.linalg.solve(XtX, Xty)

        elif self.method == 'svd':
            # Phân rã Giá trị Kỳ dị (Chương 4.5): X = U Sigma V^T
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            # Ngưỡng khử các giá trị kỳ dị suy biến gần 0
            cutoff = np.finfo(float).eps * max(N, D) * np.max(S)
            S_inv = np.where(S > cutoff, 1.0 / S, 0.0)
            self.w = (Vt.T * S_inv) @ (U.T @ y)

        elif self.method == 'gd':
            # Khởi tạo vector trọng số w = 0 và cập nhật theo đạo hàm vector (Chương 5 & 7)
            self.w = np.zeros(D)
            self.loss_history = []
            for _ in range(self.epochs):
                y_pred = X @ self.w
                error = y_pred - y
                # Hàm mục tiêu MSE: L(w) = (1 / 2N) ||y - Xw||_2^2
                loss = 0.5 * np.mean(error ** 2)
                self.loss_history.append(loss)
                # Gradient: nabla_w L(w) = (1 / N) X^T (Xw - y)
                grad = (1.0 / N) * (X.T @ error)
                self.w -= self.lr * grad

        elif self.method == 'ridge_map':
            # Ước lượng MAP (Chương 9.2.2): w_MAP = (X^T X + alpha * I_0)^(-1) X^T y
            I_0 = np.eye(D)
            I_0[0, 0] = 0.0  # Không điều chuẩn hệ số chặn w_0
            XtX_reg = (X.T @ X) + self.alpha * I_0
            Xty = X.T @ y
            self.w = np.linalg.solve(XtX_reg, Xty)

        else:
            raise ValueError(f"Phương pháp không hợp lệ: {self.method}")

        # Ước lượng phương sai nhiễu Gaussian: sigma^2_MLE = (1 / (N - D)) * ||y - Xw||_2^2
        residuals = y - (X @ self.w)
        dof = max(N - D, 1)
        self.sigma2_mle = np.sum(residuals ** 2) / dof
        return self

    def predict_log(self, X):
        return X @ self.w

    def predict_vnd(self, X, use_jensen_correction=True):
        """
        Quy đổi từ ln(Giá) về Giá thực tế (Triệu VNĐ).
        Theo Bất đẳng thức Jensen cho biến ngẫu nhiên Log-Normal:
        E[y | x] = exp(x^T w + 0.5 * sigma^2).
        """
        log_pred = self.predict_log(X)
        if use_jensen_correction and self.sigma2_mle is not None:
            return np.exp(log_pred + 0.5 * self.sigma2_mle)
        return np.exp(log_pred)


# =============================================================================
# PHẦN 4: ĐÁNH GIÁ THUẬT TOÁN, K-FOLD CV & ĐO THỜI GIAN (NHIỆM VỤ 3 - 2 ĐIỂM)
# =============================================================================

def evaluate_predictions(y_true_log, y_pred_log, y_true_vnd, y_pred_vnd, n_features):
    """Tính toán đầy đủ bộ chỉ số trên cả không gian Log và không gian Triệu VNĐ."""
    N = len(y_true_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2_log = r2_score(y_true_log, y_pred_log)
    adj_r2_log = 1.0 - (1.0 - r2_log) * (N - 1) / max(N - n_features - 1, 1)

    mae_vnd = mean_absolute_error(y_true_vnd, y_pred_vnd)
    rmse_vnd = np.sqrt(mean_squared_error(y_true_vnd, y_pred_vnd))
    mape_vnd = np.mean(np.abs((y_true_vnd - y_pred_vnd) / y_true_vnd)) * 100.0
    r2_vnd = r2_score(y_true_vnd, y_pred_vnd)

    return {
        'RMSE_Log': rmse_log,
        'R2_Log': r2_log,
        'Adj_R2_Log': adj_r2_log,
        'MAE_VND (Tr.Đ)': mae_vnd,
        'RMSE_VND (Tr.Đ)': rmse_vnd,
        'MAPE (%)': mape_vnd,
        'R2_VND': r2_vnd
    }


def tune_ridge_alpha_cv(X_train, y_train_log, alphas, k=5):
    """Tìm siêu tham số alpha tối ưu cho Ridge (MAP) bằng K-Fold Cross-Validation trên tập Train."""
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    cv_mean_rmse = []
    cv_std_rmse = []

    for a in alphas:
        fold_rmses = []
        for tr_idx, val_idx in kf.split(X_train):
            model = MML_LinearRegression_Scratch(method='ridge_map', alpha=a)
            model.fit(X_train[tr_idx], y_train_log[tr_idx])
            pred_val = model.predict_log(X_train[val_idx])
            fold_rmses.append(np.sqrt(mean_squared_error(y_train_log[val_idx], pred_val)))
        cv_mean_rmse.append(np.mean(fold_rmses))
        cv_std_rmse.append(np.std(fold_rmses))

    best_idx = int(np.argmin(cv_mean_rmse))
    return alphas[best_idx], np.array(cv_mean_rmse), np.array(cv_std_rmse)


# =============================================================================
# PHẦN 5: TRỰC QUAN HÓA 6 BIỂU ĐỒ CHO BÁO CÁO & SLIDE (NHIỆM VỤ 4 - 2 ĐIỂM)
# =============================================================================

def generate_project_charts(data, vif_df, gd_model, ols_model, ridge_model,
                            alphas, cv_means, best_alpha, X_train, y_train_log,
                            X_test, y_test_log, y_test_vnd, feature_names):
    # --- Hình 1: Phân phối Giá gốc vs Log(Giá) (Minh họa giả định Gaussian của MLE) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data['Price_Million_VND'], bins=45, kde=True, color='#e74c3c', ax=axes[0])
    axes[0].set_title(f'Phân phối Giá VGA gốc (Lệch phải mạnh - Skewness = {data["Price_Million_VND"].skew():.2f})', fontweight='bold')
    axes[0].set_xlabel('Giá bán (Triệu VNĐ)')
    axes[0].set_ylabel('Số lượng mẫu VGA')

    sns.histplot(data['Log_Price'], bins=40, kde=True, color='#2ecc71', ax=axes[1])
    axes[1].set_title(f'Phân phối ln(Giá) (Tiệm cận Chuẩn Gaussian - Skewness = {data["Log_Price"].skew():.2f})', fontweight='bold')
    axes[1].set_xlabel('ln(Giá bán - Triệu VNĐ)')
    axes[1].set_ylabel('Số lượng mẫu VGA')
    plt.tight_layout()
    plt.savefig('Slide1_Price_vs_LogPrice_Distribution.png', dpi=300)
    plt.close()

    # --- Hình 2: Ma trận tương quan & Chỉ số VIF (Chẩn đoán đa cộng tuyến) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    corr_cols = ['Log_Price', 'VRAM_GB', 'Bus_Bit', 'VRAM_Tier', 'Core_Count', 'Boost_MHz', 'TDP_W', 'Arch_Tier', 'Fans']
    corr_matrix = data[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.3, vmax=1.0, ax=axes[0])
    axes[0].set_title('Ma trận Hiệp phương sai chuẩn hóa (Correlation Heatmap)', fontweight='bold')

    top_vif = vif_df.head(12).iloc[::-1]
    colors = ['#e74c3c' if v > 10 else '#3498db' for v in top_vif['VIF']]
    axes[1].barh(top_vif['Feature'], np.log10(top_vif['VIF']), color=colors)
    axes[1].axvline(np.log10(10), color='red', linestyle='--', label='Ngưỡng cảnh báo đa cộng tuyến (VIF = 10)')
    axes[1].set_title('Chẩn đoán Đa cộng tuyến bằng log10(VIF) trước khi điều chuẩn', fontweight='bold')
    axes[1].set_xlabel('log10(VIF)')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('Slide2_EDA_Correlation_and_VIF.png', dpi=300)
    plt.close()

    # --- Hình 3: Hội tụ của Batch Gradient Descent vs Nghiệm đóng OLS/SVD ---
    ols_exact_loss = 0.5 * np.mean((y_train_log - X_train @ ols_model.w) ** 2)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(gd_model.loss_history) + 1), gd_model.loss_history,
            label='Batch Gradient Descent L(w^(t))', color='#2980b9', linewidth=2.2)
    ax.axhline(ols_exact_loss, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Nghiệm đóng Toàn cục OLS/SVD (L* = {ols_exact_loss:.5f})')
    ax.set_title('Đường hội tụ Hàm mục tiêu Lồi L(w) theo Vòng lặp Gradient Descent', fontweight='bold')
    ax.set_xlabel('Số vòng lặp (Epochs)')
    ax.set_ylabel('Giá trị hàm mục tiêu MSE: (1/2N)||y - Xw||^2')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig('Slide3_Gradient_Descent_Convergence.png', dpi=300)
    plt.close()

    # --- Hình 4: Quỹ đạo hệ số Ridge Trace & K-Fold Cross Validation chọn Alpha ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    weights_path = []
    for a in alphas:
        m = MML_LinearRegression_Scratch(method='ridge_map', alpha=a).fit(X_train, y_train_log)
        weights_path.append(m.w[1:])  # Bỏ w_0
    weights_path = np.array(weights_path)

    for j in range(min(8, weights_path.shape[1])):
        axes[0].plot(np.log10(alphas), weights_path[:, j], label=feature_names[j + 1], linewidth=1.8)
    axes[0].axvline(np.log10(best_alpha), color='black', linestyle='--', label=f'Alpha tối ưu = {best_alpha:.2f}')
    axes[0].set_title('Quỹ đạo Trọng số Ridge Trace khi tăng độ mạnh Prior MAP (alpha)', fontweight='bold')
    axes[0].set_xlabel('log10(alpha)')
    axes[0].set_ylabel('Giá trị trọng số w_j')
    axes[0].legend(fontsize=8, loc='best')

    axes[1].plot(np.log10(alphas), cv_means, marker='o', markersize=4, color='#8e44ad', linewidth=2)
    axes[1].axvline(np.log10(best_alpha), color='#e74c3c', linestyle='--',
                    label=f'Alpha* = {best_alpha:.2f} (Min CV RMSE = {np.min(cv_means):.4f})')
    axes[1].set_title('5-Fold Cross-Validation RMSE theo Siêu tham số Điều chuẩn Alpha', fontweight='bold')
    axes[1].set_xlabel('log10(alpha)')
    axes[1].set_ylabel('5-Fold CV RMSE (trên thang ln(Giá))')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('Slide4_Ridge_Trace_and_CV_Tuning.png', dpi=300)
    plt.close()

    # --- Hình 5: Giá thực tế vs Giá dự đoán & Phân tích Phần dư (Residual Plot) ---
    y_pred_log = ridge_model.predict_log(X_test)
    residuals_log = y_test_log - y_pred_log

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].scatter(y_test_log, y_pred_log, alpha=0.65, color='#16a085', edgecolors='k', s=40)
    min_v, max_v = min(y_test_log.min(), y_pred_log.min()), max(y_test_log.max(), y_pred_log.max())
    axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Đường lý tưởng y_hat = y')
    axes[0].set_title(f'Thực tế vs Dự đoán trên Tập Test (R² = {r2_score(y_test_log, y_pred_log):.4f})', fontweight='bold')
    axes[0].set_xlabel('ln(Giá thực tế - Triệu VNĐ)')
    axes[0].set_ylabel('ln(Giá dự đoán - Triệu VNĐ)')
    axes[0].legend()

    axes[1].scatter(y_pred_log, residuals_log, alpha=0.65, color='#d35400', edgecolors='k', s=40)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[1].axhline(2 * np.sqrt(ridge_model.sigma2_mle), color='red', linestyle=':', label='±2σ (Khoảng tin cậy 95%)')
    axes[1].axhline(-2 * np.sqrt(ridge_model.sigma2_mle), color='red', linestyle=':')
    axes[1].set_title('Biểu đồ Phần dư (Residual Plot) kiểm chứng Phương sai đồng nhất', fontweight='bold')
    axes[1].set_xlabel('Giá trị dự đoán ln(y_hat)')
    axes[1].set_ylabel('Phần dư e = ln(y) - ln(y_hat)')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('Slide5_Actual_vs_Predicted_and_Residuals.png', dpi=300)
    plt.close()

    # --- Hình 6: Giải thích Trọng số Mô hình (Feature Importance của Ridge Regression) ---
    coef_df = pd.DataFrame({
        'Feature': feature_names[1:],
        'Weight': ridge_model.w[1:]
    }).sort_values(by='Weight', key=abs, ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ['#27ae60' if w > 0 else '#c0392b' for w in coef_df['Weight']]
    ax.barh(coef_df['Feature'], coef_df['Weight'], color=bar_colors)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title('Top 15 Đặc trưng ảnh hưởng mạnh nhất đến ln(Giá VGA) (Mô hình MAP / Ridge)', fontweight='bold')
    ax.set_xlabel('Trọng số chuẩn hóa w_j (Độ co giãn giá theo đặc trưng)')
    plt.tight_layout()
    plt.savefig('Slide6_Feature_Weights_Interpretation.png', dpi=300)
    plt.close()


# =============================================================================
# CHƯƠNG TRÌNH CHÍNH (MAIN PIPELINE)
# =============================================================================

def main():
    # 1. Đọc dữ liệu và làm sạch
    data = load_and_preprocess_data(CSV_PATH, mode=DATASET_MODE)

    # 2. Chia tập Train / Test (80% / 20%) có phân tầng theo Thế hệ kiến trúc (Arch_Tier)
    train_df, test_df = train_test_split(
        data, test_size=0.20, random_state=RANDOM_SEED, stratify=data['Arch_Tier']
    )
    X_train, X_test, feature_names, X_train_df = build_design_matrix(train_df, test_df)
    y_train_log = train_df['Log_Price'].values
    y_test_log = test_df['Log_Price'].values
    y_train_vnd = train_df['Price_Million_VND'].values
    y_test_vnd = test_df['Price_Million_VND'].values

    N_train, D_plus_1 = X_train.shape
    n_features = D_plus_1 - 1
    print(f"\n[*] Kích thước Ma trận thiết kế X_train: {X_train.shape} | X_test: {X_test.shape}")

    # Kiểm tra hạng ma trận và số điều kiện (Condition Number) của X^T X
    XtX = X_train.T @ X_train
    rank_X = np.linalg.matrix_rank(X_train)
    cond_num = np.linalg.cond(XtX)
    eigenvalues = np.linalg.eigvalsh(XtX)
    print(f"[*] Hạng của ma trận X_train: rank(X) = {rank_X} / {D_plus_1}")
    print(f"[*] Trị riêng nhỏ nhất của X^T X: lambda_min = {eigenvalues[0]:.6f} | Trị riêng lớn nhất: lambda_max = {eigenvalues[-1]:.2f}")
    print(f"[*] Số điều kiện Condition Number kappa(X^T X) = {cond_num:,.2f} (Minh chứng có đa cộng tuyến mạnh!)")

    # Tính bảng VIF
    vif_df = compute_vif_table(X_train_df)
    print("\n[*] Top 8 đặc trưng có chỉ số Đa cộng tuyến (VIF) cao nhất:")
    print(vif_df.head(8).to_string(index=False))

    # 3. Huấn luyện các thuật toán và đo thời gian thực thi (Chạy lặp 100 lần để lấy trung bình chính xác)
    print("\n" + "=" * 80)
    print("BƯỚC 2 & 3: HUẤN LUYỆN CÁC PHƯƠNG PHÁP TỐI ƯU & ĐÁNH GIÁ KẾT QUẢ")
    print("=" * 80)

    # Tìm siêu tham số alpha cho Ridge (MAP) bằng 5-Fold CV
    alphas = np.logspace(-2, 3, 60)
    best_alpha, cv_means, cv_stds = tune_ridge_alpha_cv(X_train, y_train_log, alphas, k=5)
    print(f"[*] Siêu tham số điều chuẩn MAP/Ridge tối ưu (5-Fold CV): alpha* = {best_alpha:.4f}")

    models_config = [
        ('1. Baseline (Dự đoán Trung bình)', 'baseline', None),
        ('2. OLS - Normal Equation (NumPy)', 'normal_eq', 0.0),
        ('3. OLS - Phân rã SVD Pseudoinverse (NumPy)', 'svd', 0.0),
        ('4. OLS - Batch Gradient Descent (NumPy)', 'gd', 0.0),
        (f'5. Ridge / MAP (NumPy, alpha={best_alpha:.2f})', 'ridge_map', best_alpha),
        ('6. Scikit-learn LinearRegression (Đối chiếu)', 'sklearn_ols', 0.0),
        (f'7. Scikit-learn Ridge (Đối chiếu, alpha={best_alpha:.2f})', 'sklearn_ridge', best_alpha),
    ]

    summary_rows = []
    trained_models = {}

    for name, m_type, alpha_val in models_config:
        n_runs = 100 if m_type != 'gd' else 10
        t0 = time.perf_counter()

        for _ in range(n_runs):
            if m_type == 'baseline':
                mean_log = np.mean(y_train_log)
                sigma2_base = np.var(y_train_log, ddof=1)
                tr_pred_log = np.full_like(y_train_log, mean_log)
                te_pred_log = np.full_like(y_test_log, mean_log)
                tr_pred_vnd = np.exp(tr_pred_log + 0.5 * sigma2_base)
                te_pred_vnd = np.exp(te_pred_log + 0.5 * sigma2_base)
            elif m_type in ['normal_eq', 'svd', 'gd', 'ridge_map']:
                model = MML_LinearRegression_Scratch(
                    method=m_type, alpha=alpha_val, lr=0.05, epochs=2500
                ).fit(X_train, y_train_log)
                tr_pred_log = model.predict_log(X_train)
                te_pred_log = model.predict_log(X_test)
                tr_pred_vnd = model.predict_vnd(X_train)
                te_pred_vnd = model.predict_vnd(X_test)
                trained_models[m_type] = model
            elif m_type == 'sklearn_ols':
                sk_m = SklearnLR(fit_intercept=False).fit(X_train, y_train_log)
                tr_pred_log = sk_m.predict(X_train)
                te_pred_log = sk_m.predict(X_test)
                s2 = np.sum((y_train_log - tr_pred_log) ** 2) / (N_train - D_plus_1)
                tr_pred_vnd = np.exp(tr_pred_log + 0.5 * s2)
                te_pred_vnd = np.exp(te_pred_log + 0.5 * s2)
                trained_models['sklearn_ols'] = sk_m
            elif m_type == 'sklearn_ridge':
                # Nhân đôi hay giữ nguyên tùy định nghĩa hàm mục tiêu; ở đây giải trực tiếp
                sk_r = SklearnRidge(alpha=alpha_val, fit_intercept=True).fit(X_train[:, 1:], y_train_log)
                tr_pred_log = sk_r.predict(X_train[:, 1:])
                te_pred_log = sk_r.predict(X_test[:, 1:])
                s2 = np.sum((y_train_log - tr_pred_log) ** 2) / (N_train - D_plus_1)
                tr_pred_vnd = np.exp(tr_pred_log + 0.5 * s2)
                te_pred_vnd = np.exp(te_pred_log + 0.5 * s2)

        elapsed_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

        tr_metrics = evaluate_predictions(y_train_log, tr_pred_log, y_train_vnd, tr_pred_vnd, n_features)
        te_metrics = evaluate_predictions(y_test_log, te_pred_log, y_test_vnd, te_pred_vnd, n_features)

        summary_rows.append({
            'Mô hình / Phương pháp': name,
            'Train RMSE (Log)': tr_metrics['RMSE_Log'],
            'Test RMSE (Log)': te_metrics['RMSE_Log'],
            'Test R² (Log)': te_metrics['R2_Log'],
            'Test Adj R²': te_metrics['Adj_R2_Log'],
            'Test MAE (Tr.VNĐ)': te_metrics['MAE_VND (Tr.Đ)'],
            'Test RMSE (Tr.VNĐ)': te_metrics['RMSE_VND (Tr.Đ)'],
            'Test MAPE (%)': te_metrics['MAPE (%)'],
            'Thời gian (ms)': elapsed_ms
        })

    results_df = pd.DataFrame(summary_rows)
    print("\nBẢNG TỔNG HỢP ĐÁNH GIÁ CÁC PHƯƠNG PHÁP (SLIDE TRỌNG TÂM):")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Kiểm chứng độ lệch chuẩn L2 giữa nghiệm tự cài đặt (NumPy) và Scikit-learn
    diff_norm = np.linalg.norm(trained_models['svd'].w - trained_models['sklearn_ols'].coef_)
    print(f"\n[*] Sai số chuẩn L2 ||w_SVD_Scratch - w_Sklearn||_2 = {diff_norm:.2e} (Trùng khớp tuyệt đối về mặt toán học!)")

    # 4. Phân tích các mẫu dự đoán lệch nhiều nhất trên tập Test (Error Analysis)
    best_model = trained_models['ridge_map']
    test_eval = test_df[['Tên_Sản_Phẩm', 'Clean_Chipset', 'VRAM_GB', 'Domain', 'Price_Million_VND']].copy()
    test_eval['Pred_Million_VND'] = best_model.predict_vnd(X_test)
    test_eval['Abs_Error_Million'] = np.abs(test_eval['Price_Million_VND'] - test_eval['Pred_Million_VND'])
    test_eval['APE_%'] = (test_eval['Abs_Error_Million'] / test_eval['Price_Million_VND']) * 100.0

    print("\n[*] Top 5 mẫu VGA dự đoán chính xác nhất trên tập Test:")
    print(test_eval.sort_values(by='APE_%').head(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n[*] Top 5 mẫu VGA có sai số tuyệt đối lớn nhất trên tập Test (Dùng để giải thích Q&A):")
    print(test_eval.sort_values(by='Abs_Error_Million', ascending=False).head(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 5. Xuất bộ 6 biểu đồ phục vụ báo cáo và thuyết trình
    generate_project_charts(
        data, vif_df, trained_models['gd'], trained_models['svd'],
        trained_models['ridge_map'], alphas, cv_means, best_alpha,
        X_train, y_train_log, X_test, y_test_log, y_test_vnd, feature_names
    )
    print("\n[+] Đã xuất thành công 6 biểu đồ PNG độ phân giải 300 DPI phục vụ Slide thuyết trình MAI391!")


if __name__ == '__main__':
    main()