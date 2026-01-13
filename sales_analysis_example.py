"""
电商销售数据描述性统计分析示例代码
适用于: 本科数字经济学生实训
作者: 实训智能体
用途: 演示如何使用Python进行小样本商业数据的描述性统计分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置设置
# ============================================================

# 设置中文显示（根据系统选择合适的字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================
# 数据导入和预处理
# ============================================================

def load_and_check_data(filename):
    """
    加载数据并进行初步检查
    """
    print("="*70)
    print("第一步：数据导入和初步检查")
    print("="*70)
    
    # 读取数据
    df = pd.read_csv(filename, encoding='utf-8')
    print(f"\n✓ 数据文件 '{filename}' 加载成功！")
    
    # 显示前几行
    print("\n数据前5行预览:")
    print(df.head())
    
    # 数据基本信息
    print(f"\n数据规模: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"数据列名: {list(df.columns)}")
    
    # 检查缺失值
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("\n✓ 数据完整，无缺失值")
    else:
        print("\n⚠ 发现缺失值:")
        print(missing[missing > 0])
    
    # 数据类型
    print("\n数据类型:")
    print(df.dtypes)
    
    return df

# ============================================================
# 描述性统计分析
# ============================================================

def overall_statistics(df):
    """
    计算整体描述性统计指标
    """
    print("\n" + "="*70)
    print("第二步：整体描述性统计分析")
    print("="*70)
    
    # 转换日期格式
    df['订单日期'] = pd.to_datetime(df['订单日期'])
    
    # 基本统计指标
    total_sales = df['销售额'].sum()
    order_count = len(df)
    avg_order = df['销售额'].mean()
    median_order = df['销售额'].median()
    std_order = df['销售额'].std()
    min_order = df['销售额'].min()
    max_order = df['销售额'].max()
    
    print("\n【整体销售情况】")
    print(f"数据时间范围: {df['订单日期'].min().date()} 至 {df['订单日期'].max().date()}")
    print(f"订单总数: {order_count} 笔")
    print(f"总销售额: ¥{total_sales:,.2f}")
    print(f"\n【集中趋势指标】")
    print(f"平均订单金额: ¥{avg_order:.2f}")
    print(f"销售额中位数: ¥{median_order:.2f}")
    print(f"\n【离散程度指标】")
    print(f"销售额标准差: ¥{std_order:.2f}")
    print(f"变异系数: {(std_order/avg_order)*100:.2f}%")
    print(f"最小订单金额: ¥{min_order:.2f}")
    print(f"最大订单金额: ¥{max_order:.2f}")
    print(f"极差: ¥{max_order - min_order:.2f}")
    
    # 四分位数
    q1 = df['销售额'].quantile(0.25)
    q2 = df['销售额'].quantile(0.50)
    q3 = df['销售额'].quantile(0.75)
    iqr = q3 - q1
    
    print(f"\n【四分位数】")
    print(f"Q1 (25%分位数): ¥{q1:.2f}")
    print(f"Q2 (50%分位数): ¥{q2:.2f}")
    print(f"Q3 (75%分位数): ¥{q3:.2f}")
    print(f"四分位距(IQR): ¥{iqr:.2f}")
    
    # 分布形态
    skewness = stats.skew(df['销售额'])
    kurtosis = stats.kurtosis(df['销售额'])
    
    print(f"\n【分布形态】")
    print(f"偏度(Skewness): {skewness:.3f}", end=" ")
    if abs(skewness) < 0.5:
        print("(近似对称分布)")
    elif skewness > 0:
        print("(右偏/正偏，高值较多)")
    else:
        print("(左偏/负偏，低值较多)")
    
    print(f"峰度(Kurtosis): {kurtosis:.3f}", end=" ")
    if kurtosis > 0:
        print("(尖峭分布)")
    elif kurtosis < 0:
        print("(平坦分布)")
    else:
        print("(正态分布)")
    
    return total_sales

# ============================================================
# 分组统计分析
# ============================================================

def category_analysis(df, total_sales):
    """
    按产品类别分组分析
    """
    print("\n" + "="*70)
    print("第三步：产品类别分组分析")
    print("="*70)
    
    category_summary = df.groupby('产品类别').agg({
        '销售额': 'sum',
        '订单编号': 'count',
        '销售数量': 'sum'
    }).rename(columns={'订单编号': '订单数量'})
    
    category_summary['平均订单金额'] = (
        category_summary['销售额'] / category_summary['订单数量']
    ).round(2)
    category_summary['销售额占比(%)'] = (
        category_summary['销售额'] / total_sales * 100
    ).round(2)
    
    category_summary = category_summary.sort_values('销售额', ascending=False)
    
    print("\n产品类别统计表:")
    print(category_summary)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 销售额柱状图
    category_sales = category_summary['销售额'].sort_values(ascending=False)
    axes[0].bar(category_sales.index, category_sales.values, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('产品类别', fontsize=11)
    axes[0].set_ylabel('销售额（元）', fontsize=11)
    axes[0].set_title('各产品类别销售额对比', fontsize=12, fontweight='bold')
    for i, v in enumerate(category_sales.values):
        axes[0].text(i, v + 100, f'¥{v:.0f}', ha='center', va='bottom')
    
    # 销售额占比饼图
    colors = plt.cm.Set3.colors
    axes[1].pie(category_summary['销售额'], labels=category_summary.index, 
                autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('各产品类别销售额占比', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('产品类别分析.png', dpi=300, bbox_inches='tight')
    print("\n✓ 图表已保存: 产品类别分析.png")
    plt.show()
    
    return category_summary

def region_analysis(df):
    """
    按地区分组分析
    """
    print("\n" + "="*70)
    print("第四步：客户地区分布分析")
    print("="*70)
    
    region_summary = df.groupby('客户地区').agg({
        '销售额': 'sum',
        '订单编号': 'count'
    }).rename(columns={'订单编号': '订单数量'}).sort_values('销售额', ascending=False)
    
    print("\n地区销售统计表（TOP 10）:")
    print(region_summary.head(10))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 销售额排名
    top_regions = region_summary.head(10)
    axes[0].barh(range(len(top_regions)), top_regions['销售额'], color='coral', alpha=0.8)
    axes[0].set_yticks(range(len(top_regions)))
    axes[0].set_yticklabels(top_regions.index)
    axes[0].set_xlabel('销售额（元）', fontsize=11)
    axes[0].set_title('各地区销售额排名（TOP 10）', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    
    # 订单数量
    order_by_region = df['客户地区'].value_counts().head(10)
    axes[1].bar(range(len(order_by_region)), order_by_region.values, color='lightgreen', alpha=0.8)
    axes[1].set_xticks(range(len(order_by_region)))
    axes[1].set_xticklabels(order_by_region.index, rotation=45, ha='right')
    axes[1].set_ylabel('订单数量', fontsize=11)
    axes[1].set_title('各地区订单数量（TOP 10）', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('地区分析.png', dpi=300, bbox_inches='tight')
    print("\n✓ 图表已保存: 地区分析.png")
    plt.show()

# ============================================================
# 分布分析
# ============================================================

def distribution_analysis(df):
    """
    销售额分布分析
    """
    print("\n" + "="*70)
    print("第五步：销售额分布分析")
    print("="*70)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 直方图
    axes[0, 0].hist(df['销售额'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['销售额'].mean(), color='red', linestyle='--', linewidth=2,
                       label=f'均值: ¥{df["销售额"].mean():.2f}')
    axes[0, 0].axvline(df['销售额'].median(), color='green', linestyle='--', linewidth=2,
                       label=f'中位数: ¥{df["销售额"].median():.2f}')
    axes[0, 0].set_xlabel('销售额（元）', fontsize=11)
    axes[0, 0].set_ylabel('频数', fontsize=11)
    axes[0, 0].set_title('销售额分布直方图', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 箱线图
    bp = axes[0, 1].boxplot(df['销售额'], vert=True, patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7),
                             medianprops=dict(color='red', linewidth=2),
                             whiskerprops=dict(linewidth=1.5),
                             capprops=dict(linewidth=1.5))
    axes[0, 1].set_ylabel('销售额（元）', fontsize=11)
    axes[0, 1].set_title('销售额箱线图', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 密度曲线
    df['销售额'].plot(kind='density', ax=axes[1, 0], color='purple', linewidth=2)
    axes[1, 0].set_xlabel('销售额（元）', fontsize=11)
    axes[1, 0].set_ylabel('密度', fontsize=11)
    axes[1, 0].set_title('销售额密度曲线', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Q-Q图（正态性检验）
    stats.probplot(df['销售额'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q图（正态性检验）', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('销售额分布分析.png', dpi=300, bbox_inches='tight')
    print("\n✓ 图表已保存: 销售额分布分析.png")
    plt.show()

# ============================================================
# 时间趋势分析
# ============================================================

def time_series_analysis(df):
    """
    时间序列分析
    """
    print("\n" + "="*70)
    print("第六步：时间趋势分析")
    print("="*70)
    
    # 按月统计
    df['月份'] = df['订单日期'].dt.to_period('M')
    monthly_sales = df.groupby('月份').agg({
        '销售额': 'sum',
        '订单编号': 'count'
    }).rename(columns={'订单编号': '订单数量'})
    
    monthly_sales['环比增长(%)'] = monthly_sales['销售额'].pct_change() * 100
    
    print("\n月度销售统计:")
    print(monthly_sales)
    
    # 可视化
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 销售额折线图
    color = 'tab:blue'
    ax1.set_xlabel('月份', fontsize=12)
    ax1.set_ylabel('销售额（元）', color=color, fontsize=12)
    line1 = ax1.plot(monthly_sales.index.astype(str), monthly_sales['销售额'], 
                     color=color, marker='o', linewidth=2, markersize=8, label='销售额')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(alpha=0.3)
    
    # 订单数量柱状图
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('订单数量', color=color, fontsize=12)
    bars = ax2.bar(monthly_sales.index.astype(str), monthly_sales['订单数量'], 
                   color=color, alpha=0.4, label='订单数量')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('月度销售趋势分析', fontsize=14, fontweight='bold', pad=20)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    fig.tight_layout()
    plt.savefig('时间趋势分析.png', dpi=300, bbox_inches='tight')
    print("\n✓ 图表已保存: 时间趋势分析.png")
    plt.show()

# ============================================================
# 相关性分析
# ============================================================

def correlation_analysis(df):
    """
    变量相关性分析
    """
    print("\n" + "="*70)
    print("第七步：变量相关性分析")
    print("="*70)
    
    # 计算相关系数
    correlation = df['单价'].corr(df['销售数量'])
    print(f"\n单价与销售数量的相关系数: {correlation:.3f}")
    
    if abs(correlation) < 0.3:
        strength = "弱相关"
    elif abs(correlation) < 0.7:
        strength = "中等相关"
    else:
        strength = "强相关"
    
    direction = "正相关" if correlation > 0 else "负相关"
    print(f"相关性强度: {strength}，方向: {direction}")
    
    # 散点图
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['单价'], df['销售数量'], 
                         c=df['销售额'], cmap='viridis', 
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.xlabel('单价（元）', fontsize=12)
    plt.ylabel('销售数量', fontsize=12)
    plt.title(f'产品单价与销售数量关系散点图\n(相关系数: {correlation:.3f})', 
              fontsize=13, fontweight='bold')
    
    # 添加趋势线
    z = np.polyfit(df['单价'], df['销售数量'], 1)
    p = np.poly1d(z)
    plt.plot(df['单价'].sort_values(), p(df['单价'].sort_values()), 
             "r--", linewidth=2, label='趋势线')
    
    cbar = plt.colorbar(scatter, label='销售额（元）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('相关性分析.png', dpi=300, bbox_inches='tight')
    print("\n✓ 图表已保存: 相关性分析.png")
    plt.show()

# ============================================================
# 主程序
# ============================================================

def main():
    """
    主程序：执行完整的数据分析流程
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "电商销售数据描述性统计分析" + " "*17 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    # 1. 数据导入
    df = load_and_check_data('sample_data_sales.csv')
    
    # 2. 整体统计
    total_sales = overall_statistics(df)
    
    # 3. 分类分析
    category_summary = category_analysis(df, total_sales)
    
    # 4. 地区分析
    region_analysis(df)
    
    # 5. 分布分析
    distribution_analysis(df)
    
    # 6. 时间趋势
    time_series_analysis(df)
    
    # 7. 相关性分析
    correlation_analysis(df)
    
    # 分析完成
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)
    print("\n生成的图表文件:")
    print("  - 产品类别分析.png")
    print("  - 地区分析.png")
    print("  - 销售额分布分析.png")
    print("  - 时间趋势分析.png")
    print("  - 相关性分析.png")
    print("\n建议下一步:")
    print("  1. 查看生成的图表，分析数据特征")
    print("  2. 结合业务背景，解释分析结果")
    print("  3. 撰写分析报告，提出改进建议")
    print("  4. 尝试修改代码，进行更深入的分析")
    print("\n祝学习愉快！\n")

# ============================================================
# 运行程序
# ============================================================

if __name__ == "__main__":
    main()
