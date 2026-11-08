"""Generate a Chinese research summary for collaborators."""
from __future__ import annotations

import json

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

from .common import OUTPUTS, RESULTS, ensure_dirs


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def main() -> None:
    ensure_dirs()
    development = {
        record["model"]: record
        for record in [
            load_json(path)
            for path in RESULTS.glob("metrics_development_*.json")
        ]
    }
    external: dict[str, dict[str, dict]] = {}
    for path in RESULTS.glob("metrics_external_*.json"):
        record = load_json(path)
        external.setdefault(record["external"], {})[record["model"]] = record
    interval = load_json(RESULTS / "interval_model_analysis.json")
    holdout = load_json(RESULTS / "temporal_holdout_validation.json")
    nested = load_json(RESULTS / "nested_selection_and_utility.json")
    pilot = load_json(RESULTS / "manual_endpoint_review_pilot.json")
    review_summary = load_json(RESULTS / "manual_endpoint_review_summary.json")
    eicu_audit = load_json(RESULTS / "endpoint_feasibility_eicu_anycad.json")
    audit = eicu_audit["audits"][0]

    def metric(model: str, field: str = "auroc", digits: int = 3) -> str:
        return fmt(development.get(model, {}).get("metrics", {}).get(field), digits)

    def ext_metric(cohort: str, model: str, field: str = "auroc") -> str:
        return fmt(
            external.get(cohort, {})
            .get(model, {})
            .get("metrics", {})
            .get("unrecalibrated", {})
            .get(field)
        )

    expand = external.get("eicu_anycad", {}).get("xgb_static_ts", {})
    expand_n = int(expand.get("n", 0))
    expand_events = int(expand.get("events", 0))

    lines = [
        "# 连续时间器官图与互补时间融合：诊断编码 MACCE 三库转运研究",
        "",
        "## 一、研究定位",
        "",
        (
            "本研究评估 24 小时 ICU 连续时间生理轨迹对 7 天 MACCE 的预测价值。"
            "MIMIC-IV 用于开发，MIMIC-III 用于同机构跨时代转运，扩展 eICU "
            "用于多医院转运。主终点是诊断编码 MACCE 代理，并非已完成临床裁决"
            "的真实发病时间终点。"
        ),
        "",
        "## 二、主要结论",
        "",
        (
            f"1. 静态 XGBoost 的 AUROC 为 {metric('xgb_static')}，加入 24 小时"
            f"时间摘要后升至 {metric('xgb_static_ts')}，说明时间轨迹包含静态变量"
            "未覆盖的信息。"
        ),
        (
            f"2. XGB+TS 与三种子 Transformer 的等权时间融合达到 "
            f"{metric('temporal_ensemble')}。开发集相对 XGB+TS 提高 0.0102"
            "（95% CI 0.0077-0.0127），达到原五折分析的 0.01 材料阈值。"
        ),
        (
            f"3. 2020-2022 锁定时间留出集包含 2,456 人和 1,072 事件。时间融合"
            f"AUROC 为 {fmt(holdout['models']['temporal_ensemble']['metrics']['auroc'])}，"
            f"XGB+TS 为 {fmt(holdout['models']['xgb_static_ts']['metrics']['auroc'])}，"
            f"提高 {fmt(holdout['ensemble_minus_xgb_static_ts']['estimate'], 4)}"
            "（95% CI 0.0052-0.0166），正结果在该留出时代中复现。"
        ),
        (
            "3b. 二次强化只使用 2008-2016 训练、2017-2019 做内部时代选模，"
            "加入器官不良斜率、恢复方向、范围、临床比值和器官观察比例后，"
            "锁定 75% 增强 XGBoost + 25% Transformer。2020-2022 测试 AUROC "
            "0.7462，对照原 XGB+TS 0.7337，提高 0.0125"
            "（95% CI 0.0073-0.0172），达到材料阈值。该融合仍属于探索性模型，"
            "但特征和权重在最终时代测试前已冻结。"
        ),
        (
            "4. 严格嵌套选择下，XGB+TS 的超参数在每个外层训练折内用三折内层验证"
            f"选择。嵌套时间融合 AUROC 为 "
            f"{fmt(nested['development']['nested_temporal_ensemble']['metrics']['auroc'])}，"
            f"相对原 XGB+TS 提高 "
            f"{fmt(nested['contrasts']['nested_ensemble_minus_standard_xgb']['estimate'], 4)}"
            "（95% CI 0.0071-0.0124）。置信区间排除零，但点估计比 0.01 "
            "低约 0.0002，因此必须表述为一致但边缘，不能宣称严格嵌套后仍跨阈值。"
        ),
        (
            "4b. 进一步把 Transformer 的隐藏层、dropout 和学习率也放入嵌套"
            "选择后，联合嵌套融合 AUROC 0.7611，相对标准 XGB+TS 提高 0.0073"
            "（95% CI 0.0043-0.0106）。方向为正，但仍低于 0.01 材料阈值。"
        ),
        (
            f"5. 原动态器官图 AUROC 为 {metric('ct_graph')}，完整 CT-OGT 为 "
            f"{metric('ctogt')}，七器官 token 重构图降至 {metric('organ_graph')}。"
            "器官图假设未获支持，且不能把重构称为成功补救。"
        ),
        "",
        "## 三、队列与 eICU 扩量",
        "",
        "- MIMIC-IV：13,131 人，4,475 个诊断编码 MACCE 事件。",
        "- MIMIC-III：3,470 人，409 个事件。",
        (
            f"- 扩展 eICU：纳入任意诊断位置的 CAD，24 小时存活且至少 6 个器官"
            f"通道可观察。最终 {expand_n:,} 人、{expand_events:,} 个事件、"
            f"{audit['sites_n']} 个站点，事件率 10.9%。"
        ),
        (
            f"- 扩量前审计：首次 ICU 住院 {audit['first_stay_patients']:,} 人，"
            f"24 小时存活 {audit['eligible_24h_n']:,} 人，按时点可判定 MACCE "
            f"{audit['timed_macce_n']:,} 例；{audit['sites_ge_30_n']} 个站点"
            "在扩量前至少有 30 人和至少 1 个事件。"
        ),
        "- 原主诊断 eICU 队列（903 人、57 事件）保留为敏感性分析。",
        "",
        "## 四、外部转运",
        "",
        "| 队列 | 模型 | AUROC | 重校准后 AUROC |",
        "|---|---|---:|---:|",
    ]
    for cohort, label in (
        ("mimic3", "MIMIC-III"),
        ("eicu_anycad", "扩展 eICU"),
        ("eicu", "原 eICU 敏感性"),
    ):
        for model, model_label in (
            ("xgb_static_ts", "XGB+TS"),
            ("transformer", "Transformer"),
            ("temporal_ensemble", "时间融合"),
        ):
            record = external.get(cohort, {}).get(model, {})
            if not record:
                continue
            raw = record["metrics"]["unrecalibrated"]
            recal = record["metrics"]["recalibrated"]
            lines.append(
                f"| {label} | {model_label} | {raw['auroc']:.3f} | "
                f"{recal['auroc']:.3f} |"
            )
    lines.extend(
        [
            "",
            (
                "扩展 eICU 中时间融合相对 XGB+TS 提高 0.0107"
                "（95% CI 0.0001-0.0213）。方向一致，但下界几乎贴零，"
                "属于弱到中等证据。MIMIC-III 的配对提高为 0.0094"
                "（95% CI -0.00003 至 0.0186），跨零边界。"
            ),
            (
                "扩展 eICU 另做院际留出：增强 XGBoost 在 5 折中训练医院与"
                "测试医院完全不重叠，160 个站点总体 AUROC 0.683、AUPRC 0.233、"
                "ECE 0.010。该结果优于跨库直接转运，但属于本地跨医院学习，"
                "不能写成 MIMIC→eICU 直接转运。"
            ),
            (
                "增强 XGBoost 直接转运到 MIMIC-III 的 AUROC 为 0.670，"
                "固定 75/25 融合为 0.671；扩展 eICU 直接转运融合为 0.663，"
                "与原时间融合接近。MIMIC-III 固定融合相对原时间融合提高 "
                "0.0120（95% CI 0.0058-0.0179），扩展 eICU 差异 "
                "-0.0003（95% CI -0.0063 至 0.0059）。"
            ),
            "",
            "## 五、终点人工核查与时间区间",
            "",
            (
                "- MIMIC-IV 编码事件 4,475 例；仅 45 例满足完整时间区间都落在"
                "24-168 小时的强确认层，低于 150 事件推断阈值，因此不报告"
                "该层 AUROC。"
            ),
            (
                "- 有具体临床、操作、影像、肌钙蛋白或诊断录入锚点且区间与目标"
                "窗口重叠者 1,265 例；重训 XGB+TS AUROC 0.795。这是区间敏感性"
                "终点，不是临床真实发病时间金标准。"
            ),
            (
                f"- MIMIC-III 锚点重叠 76 例，AUROC "
                f"{fmt(interval['external']['mimic3']['macce_interval_overlap']['auroc'])}；"
                f"扩展 eICU 锚点重叠 119 例，AUROC "
                f"{fmt(interval['external']['eicu_anycad']['macce_interval_overlap']['auroc'])}；"
                "原 eICU 仅 9 例，不能做模型推断。"
            ),
            (
                "- 肌钙蛋白阈值乘数 0.5/1.0/2.0 仅使 MIMIC-IV 锚点重叠事件"
                "从 1,266 变为 1,265 和 1,254，时间融合 AUROC 均为 0.769。"
                "eICU 文档延迟 0/3/6/12 小时均为 119 个事件、AUROC 0.653，"
                "说明主区间结论对这些经验规则不敏感。"
            ),
            (
                f"- 已生成 {review_summary['rows']} 条人工核查包："
                f"{review_summary['pending']} 条 MIMIC 待独立临床核查，"
                f"{review_summary['eicu_source_text_unavailable']} 条 eICU "
                "因当前数据缺病历文本无法核查。"
            ),
            (
                f"- 完成 38 条 AI 辅助结构化图表核查试点：确认目标 "
                f"{pilot['label_counts']['confirmed_target']} 条，可能目标 "
                f"{pilot['label_counts']['probable_target']} 条，窗口前 "
                f"{pilot['label_counts']['pre_landmark']} 条，历史或非事件 "
                f"{pilot['label_counts']['historical_or_not_event']} 条，窗口后 "
                f"{pilot['label_counts']['after_horizon']} 条，信息不足 "
                f"{pilot['label_counts']['uncertain_or_insufficient']} 条。"
            ),
            (
                "- 目标窗口支持率 10/38（26.3%）。这是 AI 辅助结构化提取，"
                "不是独立临床医师裁决，不能报告为人工 PPV 金标准。"
            ),
            (
                "- 已补双读者结构化核查字段，可自动计算 PPV、时间误分类和"
                "Cohen kappa；当前 100 条 MIMIC 记录仍待独立临床医师完成，"
                "未填写的数值不会补造。"
            ),
            "",
            "## 六、临床效用置信区间",
            "",
            (
                "对 0.05-0.30 的决策阈值计算净获益并做 bootstrap。开发集与"
                "扩展 eICU 在低阈值有正净获益，但较高阈值下置信区间跨零或转负。"
                "因此临床效用依赖阈值和患病率，AUROC 增益不能直接等同于"
                "临床部署价值。"
            ),
            (
                "扩展 eICU 时间融合重校准后 Brier 从 0.176（95% CI "
                "0.173-0.179）降至 0.092（0.087-0.098），ECE 从 0.279 降至 "
                "0.005；10% 阈值下敏感度 0.601、特异度 0.614、PPV 0.159、"
                "NPV 0.927。15% 阈值下重校准相对原始概率的净获益增加 "
                "0.056-0.071。"
            ),
            "",
            "## 七、器官图重构判断",
            "",
            (
                "七器官 token 图把 20 个通道聚合为神经、循环、呼吸、肾脏、"
                "凝血、炎症和代谢七个节点，加入动态器官邻接、Time2Vec、"
                "Transformer 时间编码和静态上下文。五折 AUROC 为 0.750，"
                "低于 XGB+TS 的 0.754、原动态图的 0.756 和完整 CT-OGT 的 0.756。"
                "因此阴性结果不太可能仅由通道节点过密解释，器官图应保留为"
                "机制阴性结果，而非继续堆叠复杂性。"
            ),
            (
                "进一步加入逐小时器官变化率和观察间隔后，2017-2019 内部"
                "验证 AUROC 仅 0.724，低于预先设置的 0.738 门槛，因此该模型"
                "没有进入 2020-2022，避免在最终时代反复救图。"
            ),
            "",
            "## 八、当前证据闭环",
            "",
            (
                "已完成的闭环包括：原五折开发、锁定时间留出、嵌套模型选择、"
                "配对 bootstrap、外部三库转运、扩展 eICU 事件数、校准、"
                "决策曲线 CI、区间终点和图表核查试点。仍未闭环的是独立临床"
                "终点裁决以及前瞻性外部验证；二者不能在当前数据上伪造。"
            ),
            "",
            "## 九、可发表表述",
            "",
            (
                "最强、最稳妥的正结果是：24 小时时间信息稳定优于静态严重度，"
                "互补时间融合在开发集和锁定时间留出中改善判别，原有模型在"
                "严格嵌套后略低于材料阈值，而二代时代锁定模型达到 +0.0125。"
                "器官图没有提供额外材料增益。论文应围绕"
                "可靠时间基线、边缘融合增益和端点不确定性组织，而不是"
                "宣称器官图成功。"
            ),
            "",
            "## 十、交付物",
            "",
            "- 主稿 DOCX 与 Markdown",
            "- 补充材料 DOCX",
            "- 主表 1-8 与图 1-7",
            "- 每折预测、注意力、区间终点、人工核查包与复现实验脚本",
        ]
    )

    markdown = OUTPUTS / "CTOGT_MACCE_执行摘要_中文.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")

    document = Document()
    document.styles["Normal"].font.name = "SimSun"
    document.styles["Normal"].font.size = Pt(10.5)
    document.sections[0].left_margin = Inches(0.8)
    document.sections[0].right_margin = Inches(0.8)
    document.add_heading("连续时间器官图与互补时间融合", level=0)
    for line in lines[2:]:
        if line.startswith("## "):
            document.add_heading(line[3:], level=1)
        elif line.startswith("|"):
            continue
        elif line.strip():
            document.add_paragraph(line)

    document.add_heading("关键开发结果", level=1)
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    for index, header in enumerate(("模型", "AUROC", "AUPRC", "Brier")):
        table.rows[0].cells[index].text = header
    for model, label in (
        ("xgb_static", "XGBoost 静态"),
        ("xgb_static_ts", "XGB+TS"),
        ("transformer", "Transformer"),
        ("ct_graph", "动态器官图"),
        ("organ_graph", "七器官 token 图"),
        ("ctogt", "完整 CT-OGT"),
        ("temporal_ensemble", "时间融合"),
    ):
        record = development.get(model)
        if not record:
            continue
        cells = table.add_row().cells
        metrics = record["metrics"]
        cells[0].text = label
        cells[1].text = f"{metrics['auroc']:.3f}"
        cells[2].text = f"{metrics['auprc']:.3f}"
        cells[3].text = f"{metrics['brier']:.3f}"

    output = OUTPUTS / "CTOGT_MACCE_执行摘要_中文.docx"
    document.save(output)
    print(output)


if __name__ == "__main__":
    main()
