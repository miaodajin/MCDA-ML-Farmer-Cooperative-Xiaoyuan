import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')  # 忽略警告信息

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 50)

def score_chronic_disease(x):
    """慢性病评分：无慢性病=1，1种=0.5，2种及以上=0"""
    if pd.isna(x) or str(x).strip() == '':
        return 0
    if '11' in str(x).split(','):  # 无慢性病
        return 1
    disease_count = len(str(x).split(','))
    if disease_count == 1:
        return 0.5
    return 0

def score_health_checkup(x):
    """健康检查评分：参与集体体检=1，主动体检=0.5，不体检=0"""
    if pd.isna(x) or str(x) in ['9999', '9999.0']:
        return 0
    try:
        x = int(float(str(x).split(',')[0]))
        if x == 2:  # 参与集体组织体检
            return 1
        elif x == 1:  # 主动体检
            return 0.5
        return 0
    except:
        return 0

def score_health_knowledge(x):
    """健康知识学习评分：1=1分，0=0分"""
    if pd.isna(x) or x == 3:
        return 0
    return 1 if x == 1 else 0

def score_diet_control(row):
    """饮食控制综合评分：满足三项=1分，两项=0.67分，一项=0.33分，零项=0分"""
    count = 0
    for var in ['var20_13', 'var20_14', 'var20_15']:
        if pd.notna(row.get(var)) and row.get(var) == 1:
            count += 1
    if count == 3:
        return 1
    elif count == 2:
        return 0.67
    elif count == 1:
        return 0.33
    return 0

def score_water_source(x):
    """饮用水来源评分：自来水=1，其他=0.5，无安全水源=0"""
    if pd.isna(x):
        return 0
    x_str = str(x)
    if '1' in x_str.split(','):  # 自来水
        return 1
    if any(str(i) in x_str.split(',') for i in ['2', '3', '4', '5', '6']):  # 其他水源
        return 0.5
    return 0

def score_water_quantity(x):
    """水量充足性评分"""
    if pd.isna(x):
        return 0
    return 1 if str(x).strip() == '是' else 0

def score_waste_sorting(x):
    """垃圾分类评分"""
    if pd.isna(x):
        return 0
    return 1 if str(x).strip() == '是' else 0

def score_unified_waste(x):
    """统一处理评分"""
    if pd.isna(x):
        return 0
    return 1 if str(x).strip() == '是' else 0

def score_waste_satisfaction(x):
    """垃圾处理满意度评分：非常满意=1，比较满意=0.75，一般=0.5，不太满意=0.25，很不满意=0"""
    if pd.isna(x):
        return 0
    satisfaction_map = {
        '非常满意': 1,
        '比较满意': 0.75,
        '一般': 0.5,
        '不太满意': 0.25,
        '很不满意': 0
    }
    return satisfaction_map.get(str(x).strip(), 0)

def score_harmless_toilet(x):
    """无害化厕所评分"""
    if pd.isna(x):
        return 0
    return 1 if str(x).strip() == '是' else 0

def calculate_health_index(df_main, df_env):
    """计算健康指数"""
    # 确保varID类型一致
    df_main['varID'] = df_main['varID'].astype(str)
    df_env['varID'] = df_env['varID'].astype(str)
    
    # 计算个人健康指标得分
    scores = pd.DataFrame(index=df_main.index)
    scores['score_chronic'] = df_main['var20_8'].apply(score_chronic_disease)
    scores['score_checkup'] = df_main['var20_11'].apply(score_health_checkup)
    scores['score_knowledge'] = df_main['var20_12'].apply(score_health_knowledge)
    scores['score_diet'] = df_main.apply(score_diet_control, axis=1)
    
    # 计算环境健康指标得分
    scores['score_water'] = df_env['var10_1'].apply(score_water_source)
    scores['score_water_qty'] = df_env['var10_3'].apply(score_water_quantity)
    scores['score_waste_sort'] = df_env['var10_15'].apply(score_waste_sorting)
    scores['score_waste_unified'] = df_env['var10_16'].apply(score_unified_waste)
    scores['score_waste_satis'] = df_env['var10_17'].apply(score_waste_satisfaction)
    scores['score_toilet'] = df_env['var10_18'].apply(score_harmless_toilet)
    
    # 计算总分（所有指标权重相等）
    score_columns = [col for col in scores.columns if col.startswith('score_')]
    scores['Health_Index'] = scores[score_columns].mean(axis=1)
    
    return scores

def analyze_health_scores(scores, df_main):
    """Analyze health indicator scores and output standardized table"""
    # Prepare membership data
    membership_map = {
        '是': 1, '1': 1, 1: 1, '1.0': 1, 1.0: 1,
        '否': 0, '0': 0, 0: 0, '0.0': 0, 0.0: 0
    }
    scores['is_member'] = df_main['var5_1'].astype(str).map(membership_map).fillna(0)
    
    # Prepare results list
    results_list = []
    score_columns = [col for col in scores.columns if col.startswith('score_')] + ['Health_Index']
    
    for col in score_columns:
        # Member group data
        member_data = scores[scores['is_member'] == 1][col].dropna()
        non_member_data = scores[scores['is_member'] == 0][col].dropna()
        
        # Calculate statistics
        mean_m = member_data.mean()
        std_m = member_data.std()
        n_m = len(member_data)
        mean_nm = non_member_data.mean()
        std_nm = non_member_data.std()
        n_nm = len(non_member_data)
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(
                member_data,
                non_member_data,
                equal_var=False
            )
        except:
            t_stat, p_value = np.nan, np.nan
            
        # Determine significance
        sig = ""
        if pd.notna(p_value):
            if p_value < 0.001: sig = "***"
            elif p_value < 0.01: sig = "**"
            elif p_value < 0.05: sig = "*"
            
        # Add to results
        results_list.append({
            'Indicator': col.replace('score_', '').replace('_norm', ''),
            'Member Mean': mean_m,
            'Member Std': std_m,
            'Member N': n_m,
            'Non-Member Mean': mean_nm,
            'Non-Member Std': std_nm,
            'Non-Member N': n_nm,
            't-statistic': t_stat,
            'p-value': p_value,
            'Sig.': sig
        })

    # Sort results alphabetically by Indicator
    results_list.sort(key=lambda x: x['Indicator'])

    # Print formatted table
    print("\n--- Component Score Analysis by Cooperative Membership ---")
    separator = "=" * 110
    print(separator)
    
    # Define headers and their widths
    headers = ["Indicator", "Member Mean", "Member Std", "Member N", 
              "Non-Member Mean", "Non-Member Std", "Non-Member N", 
              "t-statistic", "p-value", "Sig."]
    
    widths = {
        "Indicator": 20,
        "Member Mean": 12,
        "Member Std": 11,
        "Member N": 9,
        "Non-Member Mean": 15,
        "Non-Member Std": 14,
        "Non-Member N": 12,
        "t-statistic": 11,
        "p-value": 8,
        "Sig.": 4
    }
    
    # Print header
    header_line = "  ".join(f"{h:<{widths[h]}}" for h in headers)
    print(header_line)
    
    # Print results
    for r in results_list:
        row_str = (
            f"{r['Indicator']:<{widths['Indicator']}}  "
            f"{r['Member Mean']:>{widths['Member Mean']-1}.3f}  "
            f"{r['Member Std']:>{widths['Member Std']-1}.3f}  "
            f"{r['Member N']:>{widths['Member N']}}  "
            f"{r['Non-Member Mean']:>{widths['Non-Member Mean']-1}.3f}  "
            f"{r['Non-Member Std']:>{widths['Non-Member Std']-1}.3f}  "
            f"{r['Non-Member N']:>{widths['Non-Member N']}}  "
            f"{r['t-statistic']:>{widths['t-statistic']-1}.3f}  "
            f"{r['p-value']:>{widths['p-value']-1}.3f}  "
            f"{r['Sig.']:<{widths['Sig.']}}"
        )
        print(row_str)
    
    print(separator)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    
    return results_list

def main():
    try:
        # 尝试不同的编码方式读取数据
        print("正在读取数据文件...")
        encodings = ['utf-8', 'gb18030', 'gbk', 'gb2312', 'latin1']
        df_main = None
        df_env = None
        
        for encoding in encodings:
            try:
                print(f"尝试使用 {encoding} 编码读取...")
                df_main = pd.read_csv('zongbiao.csv', encoding=encoding, low_memory=False)
                df_env = pd.read_csv('hh_10.csv', encoding=encoding, low_memory=False)
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                print(f"{encoding} 编码读取失败，尝试下一种编码...")
                continue
            except Exception as e:
                print(f"读取时发生其他错误：{str(e)}")
                continue
        
        if df_main is None or df_env is None:
            print("错误：无法使用任何编码方式成功读取文件")
            return

        print(f"主表行数：{len(df_main)}, 环境表行数：{len(df_env)}")
        
        # 确保关键列存在
        required_cols_main = ['varID', 'var20_8', 'var20_11', 'var20_12', 'var5_1']
        required_cols_env = ['varID', 'var10_1', 'var10_3', 'var10_15', 'var10_16', 'var10_17', 'var10_18']
        
        missing_cols_main = [col for col in required_cols_main if col not in df_main.columns]
        missing_cols_env = [col for col in required_cols_env if col not in df_env.columns]
        
        if missing_cols_main or missing_cols_env:
            print("错误：缺少必需的列：")
            if missing_cols_main:
                print(f"主表缺少：{', '.join(missing_cols_main)}")
            if missing_cols_env:
                print(f"环境表缺少：{', '.join(missing_cols_env)}")
            return
            
        # 计算健康指数
        print("正在计算健康指数...")
        scores = calculate_health_index(df_main, df_env)
        
        if scores.empty:
            print("错误：计算得分结果为空")
            return
            
        # 分析结果
        analyze_health_scores(scores, df_main)
        
    except Exception as e:
        print(f"\n分析过程中发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    try:
        print("开始健康指数详细分析...")
        main()
    except Exception as e:
        print("\nSkipping analysis because the final DataFrame could not be created.")
