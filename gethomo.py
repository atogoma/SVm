import pysam
import csv
from Bio.Seq import Seq
import sys
import multiprocessing as mp
from functools import partial
from collections import defaultdict
DEFAULT_WINDOW = 100 # read提取范围
DEFAULT_CPU_COUNT = mp.cpu_count()  # 获取CPU核心数作为默认值
   
def _normalize_chrom_name(chrom: str, bam_chroms: list) -> str:
    """标准化染色体名称以匹配BAM头"""
    # 场景1：BAM头包含'chr'前缀但输入没有
    if chrom in bam_chroms:
        return chrom
    elif f"chr{chrom}" in bam_chroms:
        return f"chr{chrom}"
    elif chrom.replace("chr", "") in bam_chroms:
        return chrom.replace("chr", "")
    else:
        raise ValueError(f"染色体名称'{chrom}'在BAM文件中不存在")

def _fetch_split_reads(bam, chrom, start, end):
    """提取指定染色体范围内[start, end)的reads"""
    fetched_reads = []
    for read in bam.fetch(chrom, start, end):
        if not read.is_unmapped and read.mapping_quality >= 20:
            fetched_reads.append(read)
    return fetched_reads

def extract_sa_positions(read, target_chr):
    """提取SA标签中的染色体、位置及CIGAR的第一个操作符，并过滤与目标染色体不匹配的记录"""
    sa_tags = read.get_tag("SA").split(";") if read.has_tag("SA") else []
    sa_info = []
    for sa in sa_tags:
        if sa:
            parts = sa.split(",")
            if len(parts) >= 4:
                chrom, pos_str, cigar = parts[0], parts[1], parts[3]
                # 解析CIGAR的第一个操作符
                op_type, op_len = None, 0
                if cigar:
                    # 分离数字和字母
                    i = 0
                    while i < len(cigar) and cigar[i].isdigit():
                        i += 1
                    if i > 0:
                        op_len = int(cigar[:i])
                        op_type = cigar[i] if i < len(cigar) else None
                # 只保留与目标染色体匹配的记录
                if chrom == target_chr:
                    sa_info.append((chrom, int(pos_str), op_type, op_len))
    return sa_info

def get_breakpoint_from_sa(reads, way, target_chr, flag):
    """通过SA Tag统计断点位置（考虑CIGAR操作符）"""
    pos_counts = defaultdict(int)
    if flag == 0:
        for read in reads:
            sa_info = extract_sa_positions(read, target_chr)
            if sa_info != []:
                print(sa_info)
                print(read.query_sequence)
            for chrom, pos, op_type, op_len in sa_info:
                if chrom == target_chr:
                    # 根据CIGAR第一个操作符调整断点位置
                    if op_type == 'S' and way == 1:
                        adjusted_pos = pos  # 软剪接，断点在pos
                        pos_counts[adjusted_pos] += 1
                    if op_type == 'M' and way == -1:
                        adjusted_pos = pos + op_len  # 匹配，断点在pos + op_len
                        pos_counts[adjusted_pos] += 1
    
    else:
        for read in reads:
            sa_info = extract_sa_positions(read, target_chr)
            for chrom, pos, op_type, op_len in sa_info:
                if chrom == target_chr:
                    # 根据CIGAR第一个操作符调整断点位置
                    if op_type == 'S' and way == -1:
                        adjusted_pos = pos  # 软剪接，断点在pos
                        pos_counts[adjusted_pos] += 1
                    if op_type == 'M' and way == 1:
                        adjusted_pos = pos + op_len  # 匹配，断点在pos + op_len
                        pos_counts[adjusted_pos] += 1
                        
    if pos_counts:
        return max(pos_counts, key=pos_counts.get)
    return 0
      
  
def process_single_row(row, bam_path, header_indices, window):
    """处理单行数据的函数，用于多进程"""
    try:
        sv_type = row[header_indices["sv_type"]]
        if sv_type not in {"TRA"}:
            return None
        # 提取坐标"DEL", "INV", "DUP", 
        chr_start = row[header_indices["chr_start"]]
        chr_end = row[header_indices["chr_end"]]
        pos_start = int(row[header_indices["pos_start"]])
        pos_end = int(row[header_indices["pos_end"]])
        or_start = int(row[header_indices["or_start"]])
        or_end = int(row[header_indices["or_end"]])
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            # 标准化染色体名称
            target_chr1 = _normalize_chrom_name(chr_start, bam.references)
            target_chr2 = _normalize_chrom_name(chr_end, bam.references)
            
            reads_bp1 = _fetch_split_reads(bam, target_chr1, start=max(0, pos_start - window),end=pos_start + window)
            reads_bp2 = _fetch_split_reads(bam, target_chr2, start=max(0, pos_end - window),end=pos_end + window)
                
            flag = 0
            if sv_type == "INV":
                flag = or_start
            bp1_np = get_breakpoint_from_sa(reads_bp1, or_start, chr_end, flag)
            homolen1,homoseq1 = generate_consensus(reads_bp1, chr_end, bp1_np, 'left')
            print(chr_start,chr_end)
            print(bp1_np)
            
            bp2_np = get_breakpoint_from_sa(reads_bp2, or_end, chr_start, flag)
            homolen2,homoseq2 = generate_consensus(reads_bp2, chr_start, bp2_np, 'right')
            print(bp2_np)
            homolen = mergeh(homolen1, homolen2)            
            if homolen == homolen1:
                homoseq = homoseq1
            else:
                homoseq = homoseq2
            
            
            mechanism = getmechanism(homolen)
            return row + [homolen, homoseq, mechanism]
            
    except Exception as e:
        print(f"处理行时发生错误: {str(e)}", flush=True)
        return row + ["er", "er", "er"]

def generate_consensus(reads, target_chr, target_pos=None, way=None):
    """生成共识序列"""

    homolen = 0
    homoseq = ""
    if target_pos is None or not reads:
        return homolen, homoseq
    if way == 'left':
        for read in reads:
            if read.has_tag("SA") and len(read.cigartuples) == 2:
                sa_info = extract_sa_positions(read, target_chr)
                seq = read.query_sequence
                should_break = False
                for chrom, sa_pos, op_type, op_len in sa_info:
                    if sa_pos == target_pos:  # S

                        if read.cigartuples[1][0] == 4:#DEL,DUP
                            homolen = read.cigartuples[0][1] - op_len
                            if homolen >= 0:
                                homoseq = seq[op_len : read.cigartuples[0][1]]
                            else:
                                homoseq = seq[read.cigartuples[0][1] : op_len]
                            should_break = True
                            break 
                        
                        if read.cigartuples[0][0] == 4:#INV
                            seq = str(Seq(seq).reverse_complement())
                            homolen = read.cigartuples[1][1] - op_len
                            if homolen >= 0:
                                homoseq = seq[op_len : read.cigartuples[1][1]]
                            else:
                                homoseq = seq[read.cigartuples[1][1] : op_len]
                            should_break = True
                            break 
                if should_break:
                    break  # 退出外层循环

                        
    else:
        for read in reads:
            if read.has_tag("SA") and len(read.cigartuples) == 2:
                
                sa_info = extract_sa_positions(read, target_chr)
                seq = read.query_sequence
                should_break = False
                for chrom, sa_pos, op_type, op_len in sa_info:
                    if sa_pos + op_len == target_pos: #M
                        if read.cigartuples[0][0] == 4:#DEL,DUP
                            homolen = op_len - read.cigartuples[0][1]
                            if homolen >= 0:
                                homoseq = seq[read.cigartuples[0][1]:op_len]
                            else:
                                homoseq = seq[op_len:read.cigartuples[0][1]]
                            should_break = True
                            break
                        
                        if read.cigartuples[0][0] == 0:#INV
                            seq = str(Seq(seq).reverse_complement())
                            homolen = op_len - read.cigartuples[1][1]
                            if homolen >= 0:
                                homoseq = seq[read.cigartuples[1][1]:op_len]
                            else:
                                homoseq = seq[op_len:read.cigartuples[1][1]]
                            should_break = True
                            break
                if should_break:
                    break  # 退出外层循环


    return homolen, homoseq

def process_sv_from_tsv(bam_path: str, input_tsv: str, output_txt: str, 
                       window: int = DEFAULT_WINDOW) -> None:
    """多进程处理TSV文件"""
    try:
        with open(input_tsv, 'r') as infile:
            reader = csv.reader(infile, delimiter='\t')
            header = next(reader, None)
            rows = list(reader)
        if not header:
            raise ValueError("输入文件缺少标题行")

        # 获取列索引
        header_indices = {
            "chr_start": header.index("ChrStart"),
            "chr_end": header.index("ChrEnd"),
            "pos_start": header.index("PosStart"),
            "pos_end": header.index("PosEnd"),
            "sv_type": header.index("Type"),
            "or_start": header.index("OrientStart"),
            "or_end": header.index("OrientEnd")
        }
        # 创建进程池（根据CPU核心数调整）
        pool = mp.Pool(processes=mp.cpu_count())

        # 使用partial固定部分参数
        worker = partial(
            process_single_row,
            bam_path=bam_path,
            header_indices=header_indices,
            window=window
        )

        # 并行处理行（保持顺序）
        results = []
        chunk_size = max(1, len(rows) // (mp.cpu_count() * 2))  # 动态分块
        for result in pool.imap(worker, rows, chunksize=chunk_size):
            results.append(result)

        pool.close()
        pool.join()

        # 写入结果到TXT文件
        with open(output_txt, 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            writer.writerow(header + ["homolen", "homoseq", "mechanism"])
            for res in results:
                if res is not None:
                    writer.writerow(res)

    except Exception as e:
        print(f"文件处理错误: {str(e)}")
        sys.exit(1)

def mergeh(h1: int, h2: int) -> str:
    if h1 == 0:
        return h2
    elif h1 < -10 or h2 < -10:
        return min(h1, h2)
    elif h1 > 100 or h2 > 100:
        return max(h1, h2)
    elif h1 > 1 or h2 > 1:
        return max(h1, h2)
    return h1

def getmechanism(h: int) -> str:
    """根据homology和insertion长度判断SV机制"""
    if h > 100:
        return  "NAHR"
    elif h < -10:
        return "FoSTeS/MMBIR" 
    elif h > 1:
        return "alt-EJ"
    return "NHEJ"

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    
    bam_path, input_tsv, output_txt = sys.argv[1], sys.argv[2], sys.argv[3]
    process_sv_from_tsv(bam_path, input_tsv, output_txt)
    print(f"\n分析完成！结果已保存至: {output_txt}")