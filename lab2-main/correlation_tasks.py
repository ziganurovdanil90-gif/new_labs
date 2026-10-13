"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def _correlations(group: pd.DataFrame) -> dict[str, float]:
    """Корреляции Пирсона признаков с MRI_Count (пропуски отбрасываются попарно)."""
    return {
        feature: float(group[feature].corr(group[TARGET], method="pearson"))
        for feature in FEATURES
    }


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.path, sep="\t", na_values=["NA"])  # поле пути: см. BrainDataInput

    # Приводим к числам: если где-то остались нечисловые обозначения пропусков, они станут NaN
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    gender = df["Gender"].astype(str).str.strip().str.lower()
    female = df[gender == "female"]
    male = df[gender == "male"]

    female_corr = _correlations(female)
    male_corr = _correlations(male)

    # Максимальный |r| среди обеих групп; NaN игнорируются
    best_feature = max(
        ((f, abs(r)) for corr in (female_corr, male_corr) for f, r in corr.items() if pd.notna(r)),
        key=lambda item: item[1],
    )[0]

    return BrainCorrelationSummary(
        female_correlations=female_corr,
        male_correlations=male_corr,
        strongest_mri_feature=best_feature,
    )
    
    
    
    
    
    import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("brainsize.txt", sep="\t", na_values=["NA"])

# Scatter-графики признаков против MRI_Count с разбивкой по полу
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, feature in zip(axes, FEATURES):
    sns.scatterplot(data=df, x=feature, y=TARGET, hue="Gender", ax=ax)
    ax.set_title(f"{feature} vs {TARGET}")
plt.tight_layout()
plt.show()

# Heatmap корреляций для каждой группы
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, group) in zip(axes, df.groupby("Gender")):
    corr = group[FEATURES + [TARGET]].corr(method="pearson")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(name)
plt.tight_layout()
plt.show()