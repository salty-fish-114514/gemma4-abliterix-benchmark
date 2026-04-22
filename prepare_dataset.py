from datasets import load_dataset
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

def prepare_all_datasets(output_dir: str = "./eval_datasets"):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    
    random.seed(42)
    
    # 数据集配置: (路径, config, split, 采样比例或固定数量, 是否全量)
    # 采样比例: 0.1 = 10%, 1.0 = 100%
    configs = {
        # 大规模数据集：按10%比例采样
        "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test", 0.1, False),
        "gsm8k": ("openai/gsm8k", "main", "test", 0.1, False),
        "ifeval": ("google/ifeval", None, "train", 0.1, False),
        "ceval": ("ceval/ceval-exam", None, "val", 0.1, False),  # 各学科按10%
        "bbh": ("lukaemon/bbh", None, None, 0.1, False),  # 各任务按10%
        
        # 小数据集：全量保留 (比例=1.0)
        "gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond", "train", 1.0, True),
        "humaneval": ("openai_humaneval", None, "test", 1.0, True),
    }
    
    # 检查 HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️  Warning: HF_TOKEN 未设置，gpqa_diamond 将跳过或失败")
        print("    获取方式: https://huggingface.co/settings/tokens\n")
    
    results_summary = {}
    
    # 处理简单数据集
    for name, (ds_path, config, split, ratio, full_keep) in configs.items():
        if name in ["ceval", "bbh"]:
            continue  # 这两个单独处理
            
        print(f"Downloading {name}...")
        try:
            load_kwargs = {}
            if config:
                load_kwargs["name"] = config
            if split:
                load_kwargs["split"] = split
            if hf_token and "gpqa" in ds_path:
                load_kwargs["token"] = hf_token
            
            ds = load_dataset(ds_path, **load_kwargs)
            
            # 处理 DatasetDict
            if hasattr(ds, 'keys') and not isinstance(ds, (list, dict)):
                available_splits = list(ds.keys())
                if split and split in available_splits:
                    ds = ds[split]
                else:
                    ds = ds[available_splits[0]]
            
            # 比例采样逻辑
            total_size = len(ds)
            if full_keep or ratio >= 1.0:
                sample_size = total_size
                indices = list(range(total_size))
                print(f"  ℹ️  Full dataset: {total_size} samples (100%)")
            else:
                sample_size = max(1, int(total_size * ratio))  # 至少采1个
                indices = random.sample(range(total_size), sample_size)
                print(f"  ℹ️  Sampled: {sample_size}/{total_size} ({ratio*100:.0f}%)")
            
            sampled = [ds[i] for i in indices]
            
            # 添加元信息
            meta = {
                "_dataset_name": name,
                "_total_original": total_size,
                "_sampled_count": sample_size,
                "_sampling_ratio": ratio if not full_keep else 1.0,
                "_sampling_method": "full" if full_keep else f"random_{ratio}"
            }
            
            output_data = {
                "metadata": meta,
                "samples": sampled
            }
            
            with open(out / f"{name}_sampled.json", "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            results_summary[name] = {
                "original": total_size,
                "sampled": sample_size,
                "ratio": ratio if not full_keep else 1.0
            }
            print(f"  ✓ Saved to {out}/{name}_sampled.json")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # 处理 C-Eval: 各学科按10%比例采样
    print("\nDownloading ceval (all subjects, proportional sampling)...")
    try:
        ceval_result = sample_ceval_proportional(out, random, ratio=0.1)
        results_summary["ceval"] = ceval_result
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 处理 BBH: 各任务按10%比例采样
    print("\nDownloading bbh (all tasks, proportional sampling)...")
    try:
        bbh_result = sample_bbh_proportional(out, random, ratio=0.1)
        results_summary["bbh"] = bbh_result
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 打印汇总
    print_summary(results_summary)

def sample_ceval_proportional(out: Path, random_module, ratio: float = 0.1) -> Dict:
    """
    C-Eval 各学科按相同比例采样，保持学科间比例不变
    """
    subjects = [
        'accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine',
        'business_administration', 'chinese_language_and_literature', 'civil_servant',
        'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics',
        'college_programming', 'computer_architecture', 'computer_network',
        'discrete_mathematics', 'education_science', 'electrical_engineer',
        'environmental_impact_assessment_engineer', 'fire_engineer', 'high_school_biology',
        'high_school_chemistry', 'high_school_chinese', 'high_school_geography',
        'high_school_history', 'high_school_mathematics', 'high_school_physics',
        'high_school_politics', 'ideological_and_moral_cultivation', 'law',
        'legal_professional', 'logic', 'mao_zedong_thought', 'marxism', 'metrology_engineer',
        'middle_school_biology', 'middle_school_chemistry', 'middle_school_geography',
        'middle_school_history', 'middle_school_mathematics', 'middle_school_physics',
        'middle_school_politics', 'modern_chinese_history', 'operating_system',
        'physician', 'plant_protection', 'probability_and_statistics',
        'professional_tour_guide', 'sports_science', 'tax_accountant',
        'teacher_qualification', 'urban_and_rural_planner', 'veterinary_medicine'
    ]
    
    all_samples = []
    subject_stats = {}
    total_original = 0
    total_sampled = 0
    
    for subject in subjects:
        try:
            ds = load_dataset("ceval/ceval-exam", subject, split="val")
            subject_total = len(ds)
            subject_size = max(1, int(subject_total * ratio))  # 至少采1个
            
            indices = random_module.sample(range(subject_total), subject_size)
            samples = [ds[i] for i in indices]
            
            # 标记来源
            for s in samples:
                s['_subject'] = subject
            
            all_samples.extend(samples)
            
            subject_stats[subject] = {
                "original": subject_total,
                "sampled": subject_size,
                "ratio": subject_size / subject_total
            }
            total_original += subject_total
            total_sampled += subject_size
            
        except Exception as e:
            print(f"    ⚠️  {subject}: {e}")
    
    # 保存
    meta = {
        "_dataset_name": "ceval",
        "_total_original": total_original,
        "_sampled_count": total_sampled,
        "_sampling_ratio": ratio,
        "_sampling_method": f"proportional_{ratio}",
        "_subject_breakdown": subject_stats
    }
    
    output_data = {
        "metadata": meta,
        "samples": all_samples
    }
    
    with open(out / "ceval_sampled.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Total: {total_sampled}/{total_original} ({ratio*100:.0f}%)")
    print(f"    Subjects: {len(subject_stats)} loaded")
    
    return {
        "original": total_original,
        "sampled": total_sampled,
        "ratio": ratio
    }

def sample_bbh_proportional(out: Path, random_module, ratio: float = 0.1) -> Dict:
    """
    BBH 各任务按相同比例采样，保持任务间比例不变
    """
    tasks = [
        'boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa',
        'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton',
        'logical_deduction_five_objects', 'logical_deduction_seven_objects',
        'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two',
        'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
        'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding',
        'temporal_sequences', 'tracking_shuffled_objects_five_objects',
        'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
        'web_of_lies', 'word_sorting'
    ]
    
    all_samples = []
    task_stats = {}
    total_original = 0
    total_sampled = 0
    
    for task in tasks:
        try:
            ds = load_dataset("lukaemon/bbh", task)
            split_name = list(ds.keys())[0]
            ds = ds[split_name]
            
            task_total = len(ds)
            task_size = max(1, int(task_total * ratio))  # 至少采1个
            
            indices = random_module.sample(range(task_total), task_size)
            samples = [ds[i] for i in indices]
            
            # 标记来源
            for s in samples:
                s['_task'] = task
            
            all_samples.extend(samples)
            
            task_stats[task] = {
                "original": task_total,
                "sampled": task_size,
                "ratio": task_size / task_total
            }
            total_original += task_total
            total_sampled += task_size
            
        except Exception as e:
            print(f"    ⚠️  {task}: {e}")
    
    # 保存
    meta = {
        "_dataset_name": "bbh",
        "_total_original": total_original,
        "_sampled_count": total_sampled,
        "_sampling_ratio": ratio,
        "_sampling_method": f"proportional_{ratio}",
        "_task_breakdown": task_stats
    }
    
    output_data = {
        "metadata": meta,
        "samples": all_samples
    }
    
    with open(out / "bbh_sampled.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Total: {total_sampled}/{total_original} ({ratio*100:.0f}%)")
    print(f"    Tasks: {len(task_stats)} loaded")
    
    return {
        "original": total_original,
        "sampled": total_sampled,
        "ratio": ratio
    }

def print_summary(summary: Dict[str, Dict]):
    """打印采样汇总表"""
    print("\n" + "="*60)
    print("采样结果汇总")
    print("="*60)
    print(f"{'Dataset':<20} {'Original':<10} {'Sampled':<10} {'Ratio':<10}")
    print("-"*60)
    
    total_orig = 0
    total_samp = 0
    
    for name, stats in summary.items():
        orig = stats["original"]
        samp = stats["sampled"]
        ratio = stats["ratio"]
        ratio_str = f"{ratio*100:.0f}%" if ratio < 1.0 else "100%"
        
        print(f"{name:<20} {orig:<10} {samp:<10} {ratio_str:<10}")
        total_orig += orig
        total_samp += samp
    
    print("-"*60)
    print(f"{'TOTAL':<20} {total_orig:<10} {total_samp:<10} {total_samp/total_orig*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    prepare_all_datasets()
