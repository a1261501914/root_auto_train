"""准备用户确认的白色根系标签，并排除未标注目录中的已标注副本。"""
import json
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from root_self_train.data import digest, write_json


def main():
    parser = argparse.ArgumentParser(description='准备标签副本和固定划分，生成本机或服务器运行配置')
    parser.add_argument('--unlabeled-dir', default='G:/2025夏季欧阳老师大根盒水稻实验/2025_big_rice_org/Data1011')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    external = Path(args.unlabeled_dir).resolve()
    if not external.is_dir():
        raise FileNotFoundError(external)
    images = sorted((root/'dataset/org').glob('*.png'))
    shuffled = list(images)
    random.Random(42).shuffle(shuffled)
    val_ids = {p.stem for p in shuffled[:2]}
    output = root/'data/prepared'
    (output/'masks').mkdir(parents=True, exist_ok=True)
    pairs = {'train': [], 'val': []}
    labeled_hashes = set()
    for image in images:
        mask = root/'dataset/label'/image.name
        with Image.open(mask) as im:
            arr = np.asarray(im)
            if not set(np.unique(arr)).issubset({0,255}):
                raise ValueError(f'Expected confirmed 0/255 mask: {mask}')
            converted = (arr == 255).astype(np.uint8)
        destination = output/'masks'/image.name
        Image.fromarray(converted).save(destination)
        split = 'val' if image.stem in val_ids else 'train'
        pairs[split].append(dict(image=str(image), mask=str(destination)))
        labeled_hashes.add(digest(image))
    for split, rows in pairs.items():
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
        (output/f'{split}.txt').write_text(''.join(row['image']+'\t'+row['mask']+'\n' for row in rows),encoding='utf-8')
    selected, excluded, seen = [], [], set()
    files = sorted(p for p in external.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'})
    for index, image in enumerate(files):
        h = digest(image)
        # 同名图也保守排除，防止重编码或改动标签前后的同源图进入另一集合。
        if h in labeled_hashes or image.stem in {p.stem for p in images}:
            excluded.append(dict(image=str(image), reason='labeled_content_or_stem'))
        elif h in seen:
            excluded.append(dict(image=str(image), reason='duplicate_unlabeled'))
        else:
            selected.append(dict(image=str(image)))
            seen.add(h)
        if index % 100 == 0:
            print(f'Indexed {index+1}/{len(files)}', flush=True)
    (output/'unlabeled.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected),encoding='utf-8')
    # 冒烟测试只用前 8 张，避免首次联调就预测整个大图目录。
    (output/'unlabeled_smoke.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected[:8]),encoding='utf-8')
    write_json(output/'split_report.json', dict(seed=42, train=[p['image'] for p in pairs['train']],
        val=[p['image'] for p in pairs['val']], unlabeled=len(selected), excluded=excluded,
        label_mapping={'0':0, '255':1}, note='Small provisional split for integration; not a final test set.'))
    cfg = yaml.safe_load((root/'config.example.yaml').read_text())
    cfg.update(model_config='configs/root_segformer_b2.yml', checkpoint='root_get/model.pdparams',
        train_manifest='data/prepared/train.jsonl', val_manifest='data/prepared/val.jsonl',
        unlabeled_dir=str(external), unlabeled_manifest='data/prepared/unlabeled_smoke.jsonl',
        workdir='runs/local_smoke_001', rounds=1, iters=10, batch_size=1, device=args.device)
    (root/'config.local.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(f'Prepared train={len(pairs["train"])}, val={len(pairs["val"])}, unlabeled={len(selected)}, excluded={len(excluded)}',flush=True)


if __name__ == '__main__':
    main()
