import pandas as pd
import numpy as np
from scipy.stats import variation

# ----------------------------
# 1. 定义SV长度区间（仅适用于同染色体事件）
# ----------------------------
def get_sv_length_category(length):
    """根据SV长度划分区间（DEL/DUP/INV）"""
    if pd.isna(length) or length < 0:
        return "invalid"
    elif 1000 <= length < 10000:
        return "1-10Kb"
    elif 10000 <= length < 100000:
        return "10-100Kb"
    elif 100000 <= length < 1e6:
        return "100Kb-1Mb"
    elif 1e6 <= length < 1e7:
        return "1Mb-10Mb"
    elif length >= 1e7:
        return ">10Mb"
    else:
        return "unknown"

# ----------------------------
# 2. 加载并分类SV数据
# ----------------------------
def load_and_classify_sv(sv_path):
    """加载SV数据，分类DEL/DUP/INV的区间和TRA的平衡性"""
    sv_df = pd.read_csv(sv_path, sep="\t")
    
    # 预处理：统一染色体格式并过滤非法位置
    sv_df["chrom1"] = sv_df["chrom1"].astype(str).str.replace("chr|Chr", "", regex=True)
    sv_df["chrom2"] = sv_df["chrom2"].astype(str).str.replace("chr|Chr", "", regex=True)
    
    # 仅在同染色体时计算长度，否则设为NaN
    same_chrom_mask = (sv_df["chrom1"] == sv_df["chrom2"])
    sv_df["length"] = np.where(
        same_chrom_mask,
        sv_df["pos2"] - sv_df["pos1"] + 1,
        np.nan
    )
    
    # 分类DEL/DUP/INV的长度区间
    sv_df["category"] = sv_df.apply(
        lambda row: (
            f"{row['svtype'].upper()}_{get_sv_length_category(row['length'])}"
            if row["svtype"].upper() in ["DEL", "DUP", "INV"] and pd.notna(row["length"])
            else (
                "TRA_RECIP" if (row["svtype"].upper() == "TRA" and row["strand1"] == row["strand2"])
                else "TRA_UNBAL" if (row["svtype"].upper() == "TRA" and row["strand1"] != row["strand2"])
                else "invalid"
            )
        ),
        axis=1
    )
    
    # 清理无效数据
    sv_df = sv_df[~sv_df["category"].isin(["invalid"])]
    
    return sv_df

# ----------------------------
# 3. 统计各样本的SV计数
# ----------------------------
def count_sv_events(sv_df):
    """按样本和分类统计SV事件数量"""
    # 生成所有可能的分类标签
    categories = []
    for sv_type in ["DEL", "DUP", "INV"]:
        for size_range in ["1-10Kb", "10Kb-100Kb", "100Kb-1Mb", "1Mb-10Mb", ">10Mb"]:
            categories.append(f"{sv_type}_{size_range}")
    categories += ["TRA_RECIP", "TRA_UNBAL"]
    
    # 按样本和分类统计
    counts = sv_df.groupby(["sample", "category"]).size().unstack(fill_value=0)
    
    # 确保所有分类都存在列
    for cat in categories:
        if cat not in counts.columns:
            counts[cat] = 0
    
    return counts[categories]

# ----------------------------
# 4. 计算断裂点分散评分（仅统计同染色体事件）
# ----------------------------
def calculate_dispersion(sv_df):
    """计算每个样本的断裂点位置变异系数"""
    dispersion = {}
    for sample in sv_df["sample"].unique():
        sample_sv = sv_df[(sv_df["sample"] == sample) & (sv_df["chrom1"] == sv_df["chrom2"])]
        positions = pd.concat([sample_sv["pos1"], sample_sv["pos2"]])
        dispersion[sample] = variation(positions) if len(positions) > 1 else 0
    return pd.Series(dispersion, name="dispersion_score")

# ----------------------------
# 5. 处理CNV数据并计算特征
# ----------------------------
def process_cnv(cnv_path, gender_path=None, loss_factor=0.85, gain_factor=1.15):
    """
    处理CNV数据并计算特征（支持自动推断性别）
    :param cnv_path: CNV文件路径，需包含列 ["chromosome", "start", "end", "total_cn", "sample"]
    :param gender_path: 性别文件路径（可选），需包含列 ["sample", "gender"]
    :param loss_factor: 拷贝丢失阈值因子（默认0.85）
    :param gain_factor: 拷贝增益阈值因子（默认1.15）
    :return: CNV特征矩阵，包含列 ["loss_percent", "gain_percent", "max_cn", ...]
    """
    # 加载CNV数据并预处理
    cnv = pd.read_csv(cnv_path, sep="\t")
    cnv["chromosome"] = cnv["chromosome"].astype(str).str.replace("chr|Chr", "", regex=True)
    
    # ----------------------------
    # 1. 处理性别信息（自动推断或加载外部文件）
    # ----------------------------
    if gender_path:
        # 如果提供了性别文件，直接加载
        gender = pd.read_csv(gender_path, sep="\t")
        gender["gender"] = gender["gender"].str.lower().map({
            "male": "Male", "m": "Male",
            "female": "Female", "f": "Female"
        }).fillna("Unknown")
    else:
        # 未提供性别文件时，从CNV数据推断性别
        def _infer_gender(sample_cnv):
            """辅助函数：推断单个样本的性别"""
            x_cnv = sample_cnv[sample_cnv["chromosome"] == "X"]["total_cn"]
            if x_cnv.empty:
                return "Unknown"
            x_median = x_cnv.median()
            if (1 * loss_factor) <= x_median <= (1 * gain_factor):
                return "Male"
            elif (2 * loss_factor) <= x_median <= (2 * gain_factor):
                return "Female"
            else:
                return "Unknown"
        
        # 按样本分组并推断性别
        gender = cnv.groupby("sample").apply(_infer_gender).reset_index()
        gender.columns = ["sample", "gender"]
    
    # 合并性别信息到CNV数据
    cnv = pd.merge(cnv, gender, on="sample", how="left")
    
    # ----------------------------
    # 2. 计算拷贝数基线阈值
    # ----------------------------
    cnv["baseline"] = np.where(
        (cnv["chromosome"].isin(["X", "Y"]) & 
        (cnv["gender"] == "Male"),
        1.0,  # 男性X/Y染色体基线为1
        2.0    # 其他情况基线为2
    ))
    
    # ----------------------------
    # 3. 计算拷贝丢失和增益
    # ----------------------------
    cnv["loss"] = cnv["total_cn"] < (cnv["baseline"] * loss_factor)
    cnv["gain"] = cnv["total_cn"] > (cnv["baseline"] * gain_factor)
    # 计算每个 CNV 区域的长度
    cnv["length"] = cnv["end"] - cnv["start"] + 1

    # 计算每个样本的总 CNV 长度
    total_cnv_length = cnv.groupby("sample")["length"].sum()

    # ----------------------------
    # 4. 按样本汇总统计特征
    # ----------------------------
    # 假设基因组总长度（例如hg19为3e9）
    genome_length = 3e9
    
    cnv_metrics = cnv.groupby("sample").agg(
        loss_percent=("loss", "mean"),
        gain_percent=("gain", "mean"),
        max_cn=("total_cn", "max"),
        loss_length_percent=("length", lambda x: (x[cnv["loss"]].sum() / total_cnv_length[x.name] * 100)),
        gain_length_percent=("length", lambda x: (x[cnv["gain"]].sum() / total_cnv_length[x.name] * 100))
    ).round(4)
    
    return cnv_metrics

# ----------------------------
# 主流程
# ----------------------------
if __name__ == "__main__":
    # 输入文件路径
    sv_path = "sv_input.tsv"
    cnv_path = "cnv_input.tsv"
    gender_path = "gender_input.tsv"
    
    # 1. 加载并分类SV数据
    sv_df = load_and_classify_sv(sv_path)
    
    # 2. 统计SV计数
    sv_counts = count_sv_events(sv_df)
    
    # 3. 计算断裂点分散评分
    dispersion_scores = calculate_dispersion(sv_df)
    
    # 4. 处理CNV数据
    cnv_metrics = process_cnv(cnv_path, gender_path)
    
    # 5. 合并所有特征
    feature_matrix = pd.concat([sv_counts, dispersion_scores, cnv_metrics], axis=1)
    feature_matrix.to_csv("feature_matrix.csv", index=True)