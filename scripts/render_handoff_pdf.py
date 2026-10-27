"""生成严格两页技术总结和一页迁移笔记；最终需逐页渲染检查。"""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont("Chinese", str(args.font), subfontIndex=0))
    output = root / "output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    navy, teal = colors.HexColor("#17324D"), colors.HexColor("#007C83")
    styles = {
        "title": ParagraphStyle("title", fontName="Chinese", fontSize=23, leading=32, textColor=navy,
                                spaceAfter=7, wordWrap="CJK"),
        "sub": ParagraphStyle("sub", fontName="Chinese", fontSize=9, leading=14, textColor=teal,
                              spaceAfter=14, wordWrap="CJK"),
        "h": ParagraphStyle("h", fontName="Chinese", fontSize=12, leading=18, textColor=teal,
                            spaceBefore=10, spaceAfter=5, wordWrap="CJK"),
        "body": ParagraphStyle("body", fontName="Chinese", fontSize=9.3, leading=15,
                               textColor=navy, spaceAfter=7, wordWrap="CJK"),
        "small": ParagraphStyle("small", fontName="Chinese", fontSize=8, leading=12,
                                textColor=navy, spaceAfter=5, wordWrap="CJK"),
        "cell": ParagraphStyle("cell", fontName="Chinese", fontSize=8.1, leading=12,
                               textColor=navy, wordWrap="CJK"),
    }

    def paragraph(text, style="body"):
        return Paragraph(escape(text), styles[style])

    def table(rows, widths):
        contents = [[paragraph(str(value), "cell") for value in row] for row in rows]
        result = Table(contents, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEEF0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F9")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, teal),
        ]))
        return result

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#D6DFE6"))
        canvas.line(42, 36, A4[0] - 42, 36)
        canvas.setFont("Chinese", 7.5)
        canvas.setFillColor(navy)
        canvas.drawString(42, 23, "Farfield ASR | 2026-10-28 | 实验记录与可复现工程")
        canvas.drawRightString(A4[0] - 42, 23, str(doc.page))

    story = [paragraph("多通道远场语音前端", "title"),
             paragraph("技术总结 / 1  问题、方法与主评测", "sub"),
             paragraph("目标：把增强与识别放到同一张可核验的结果表中", "h"),
             paragraph("输入含混响、点源干扰与扩散噪声的多通道语音，输出单通道音频送入固定中文 ASR。"
                       "同时报告直达参考 SI-SDR、PESQ-WB、STOI 和语料级 CER，保留效果退化的负结果。"),
             paragraph("数据与隔离", "h"),
             paragraph("原始 AISHELL 测试划分 200 条、20 位说话人，4 麦线阵/6 麦圆阵，RT60 为 0.2/0.4/0.6 秒，"
                       "多房间、距离与方向。训练 320 条、开发 80 条，各两次仿真，共 800 个样本；说话人与源噪声跨集合隔离。"
                       "固定种子，验证源文件、分量加和、SNR/SIR、前缀与产物哈希；不声称验证完整原始归档 MD5。"),
             paragraph("算法链路", "h"),
             paragraph("多通道 STFT → DAS / 已知方向 MVDR / 掩码加权协方差 Souden MVDR → ISTFT → 固定 ASR。"
                       "MVDR 解线性方程，对角加载为 1e-3 × 通道平均功率 + 1e-12；0.6 秒纯噪声前缀的完整窗估计噪声协方差。"
                       "神经分支用参考通道 log-power、两层单向 GRU、83,137 参数，预测功率 IRM。CPU 训练 20 epoch，最佳开发轮次 19。"),
             paragraph("相同 200 条主测试的实测结果", "h")]
    rows = [["前端", "SI-SDR / dB", "PESQ", "STOI", "CER"],
            ["参考通道", "-8.327", "1.064", "0.486", "37.89%"],
            ["DAS", "-6.090", "1.079", "0.524", "28.91%"],
            ["Oracle MVDR", "-2.501", "1.102", "0.616", "17.27%"],
            ["IRM MVDR", "-1.978", "1.177", "0.602", "17.41%"],
            ["神经 MVDR", "-6.176", "1.074", "0.512", "33.37%"],
            ["单通道补充模型", "-4.871", "1.164", "0.500", "51.85%"]]
    story += [table(rows, [145, 100, 80, 80, 106]), Spacer(1, 8),
              paragraph("CER = 总编辑数 / 总参考字符数；1400 条解码均成功。干净 200 条 CER 为 2.66%，固定 50 条子集为 1.75%。"
                        "ASR 为 SenseVoiceSmall int8 ONNX，固定中文、关闭 ITN、batch=1；控制 token 去除后使用统一 NFKC 文本规范化。", "small"),
              paragraph("Oracle/IRM 读取额外真值，属于诊断参考，不能冒充可部署方案或逐指标数学上界。单通道对照为本轮补建，"
                        "复用同一网络权重，未声称完整项目一 AEC/VAD 已完成。", "small"), PageBreak(),
              paragraph("结果边界与工程判断", "title"),
              paragraph("技术总结 / 2  消融、失败与复现", "sub"),
              paragraph("配对消融：改变一个因素，保留同一声场", "h"),
              paragraph("20 位测试说话人各一段最长 4 秒语音；所有阵列麦克风联合仿真，共享点噪声与扩散场。"
                        "最大孔径固定 0.12 m，SNR=5 dB/SIR=0 dB 在虚拟中心校准，指标与导向也以中心为参考。"
                        "完成 60 次场景仿真、1320 次有效评分；该协议与主测试的参考不同，不能直接混表。"),
              Image(str(root / "results/generated/phase7/doa_error.png"), width=511, height=170),
              paragraph("图：4 麦线阵、RT60=0.4 秒；20 位说话人均值与标准误。", "small"),
              paragraph("主要结论", "h"),
              paragraph("1. 固定孔径下，线阵 MVDR 从 2 到 6 麦的 SI-SDR 从 -5.550 到 -3.851 dB；6 麦圆阵为 -1.202 dB。"
                        "只说明本组场景的数量与几何收益，不能外推为所有方向的排名。"),
              paragraph("2. RT60 从 0.2 到 0.6 秒时，4 麦线阵 MVDR 的 SI-SDR 从 0.685 到 -6.376 dB。"
                        "正负 20 度导向误差的对称平均损失：MVDR 1.231 dB，DAS 0.140 dB，显示自适应滤波的失配敏感性。"),
              paragraph("3. 神经 MVDR 未超过 DAS；单通道 SI-SDR/PESQ 改善却损害 CER。测试掩码图显示预测平滑且漏估语音高占比频点，"
                        "但掩码偏差、协方差污染与 ASR 分布偏移仍需配对干预才能归因。已整理十个不同场景的文本与指标案例。"),
              paragraph("4. GCC-PHAT/SRP-PHAT/MUSIC 的目标方向 MAE 分别为 62.54/64.17/61.85 度。"
                        "三类失败谱图显示峰值接近干扰方向；算法缺少目标身份信息，不能把主导源定位当成目标说话人定位。"),
              paragraph("复现证据与下一步", "h"),
              paragraph("75 项测试、Ruff、两套环境依赖检查通过；特征、训练、增强与解码缓存绑定输入/代码/检查点哈希。"
                        "完整命令见 docs/phase5_validation.md 与 docs/phase6_7_validation.md。当前只验证仿真数据，"
                        "训练说话人覆盖较少、全句协方差为离线算法；优先扩大训练覆盖、验证目标保持策略及真实录音泛化。"),
              paragraph("来源：QwenAudio/SenseVoice 官方仓库 https://github.com/QwenAudio/SenseVoice；"
                        "数据来源与条件详见 docs/data_sources.md。所有数字为本项目实际子集结果，不宣称复现官方全量精度。", "small")]
    SimpleDocTemplate(str(output / "technical_summary.pdf"), pagesize=A4, rightMargin=42,
                      leftMargin=42, topMargin=40, bottomMargin=48,
                      title="多通道远场语音前端 - 两页技术总结").build(story, onFirstPage=footer, onLaterPages=footer)

    note = [paragraph("从水声阵列到麦克风阵列", "title"),
            paragraph("一页迁移笔记 / 共用模型、介质差异与本项目验证", "sub"),
            paragraph("可以迁移的是空间处理方法与验证习惯", "h"),
            paragraph("频域模型 X(f,t)=a(f,theta)S(f,t)+N(f,t) 提供共同起点：时差补偿、空间协方差、"
                      "约束波束形成和子空间估计仍有意义。导向符号、参考通道、阵元坐标与频率单位必须先统一，"
                      "再谈把某个公式迁移到新介质。"),
            table([["问题", "水声场景", "本项目的麦克风场景"],
                   ["传播环境", "水体、海面/海底边界及声速剖面影响传播", "房间边界反射；固定声速 343 m/s"],
                   ["非平稳性", "长多径与运动诱发的 Doppler 常需显式建模", "离线静态 RIR；没有模拟运动或 Doppler"],
                   ["阵列设计", "应按工作频率、波长、孔径与环境共同设计", "0.12 m 固定孔径消融，不能只比较麦克风数量"],
                   ["评价目标", "任务可为定位、检测或通信，指标取决于任务", "增强指标并行报告 CER，避免目标错位"]],
                  [76, 215, 220]),
            paragraph("需要重新检查的三个尺度", "h"),
            paragraph("波长 lambda=c/f 决定间距与空间混叠；传播时差约为孔径/c，影响采样与时间分辨率；"
                      "相干时间、混响长度与 STFT 窗长决定窄带/平稳近似是否合适。不能只替换声速常数就认为水声代码已适用室内语音。"),
            paragraph("把迁移落到可验证的实验", "h"),
            paragraph("先在理想平面波上检查 w^H a=1 和方向符号，再在真实 RIR 中加入混响与干扰。"
                      "本项目固定孔径、联合生成共享声场、使用共同中心参考，区分几何变化与参考延迟变化。"
                      "MVDR 的正负 20 度导向失配平均损失为 1.231 dB，提醒我们导向模型与协方差估计同样重要。"),
            paragraph("不能直接迁移的结论", "h"),
            paragraph("水声通信中的均衡/多普勒补偿效果，不等价于语音识别收益；本项目的静态房间实验也不能验证水下传播。"
                      "共享的是推导、数值稳定性与受控实验方法，不是现成的性能数字。"),
            paragraph("资料与边界", "h"),
            paragraph("Li, Chitre & Stojanovic, Underwater acoustic communications, Nature Reviews Electrical Engineering (2024). "
                      "综述支持水下多径、运动 Doppler 与传播环境差异；本页的代码迁移步骤与实验解释为本项目的工程推论。", "small"),
            paragraph("https://doi.org/10.1038/s44287-024-00122-w", "small"),
            paragraph("作者机构公开稿：https://arl.nus.edu.sg/wp-content/uploads/2025/03/Underwater-acoustic-communications.pdf", "small")]
    SimpleDocTemplate(str(output / "underwater_to_microphone.pdf"), pagesize=A4, rightMargin=42,
                      leftMargin=42, topMargin=40, bottomMargin=48,
                      title="从水声阵列到麦克风阵列 - 一页笔记").build(note, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    main()
