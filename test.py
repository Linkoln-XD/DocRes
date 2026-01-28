import torch
from collections import OrderedDict


def analyze_checkpoint(checkpoint_path):
    print(f"Анализ файла: {checkpoint_path}")

    # Загружаем checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Извлекаем state_dict
    state_dict = checkpoint['model_state']
    print(f"Найден state_dict с {len(state_dict)} параметрами")

    print("\n=== ОСНОВНОЙ АНАЛИЗ АРХИТЕКТУРЫ ===")

    # 1. Выводим ключевые параметры
    print("1. Ключевые параметры модели:")

    # Ищем все уникальные префиксы
    prefixes = set()
    for k in state_dict.keys():
        if '.' in k:
            prefix = k.split('.')[0]
            prefixes.add(prefix)

    print("Уникальные префиксы:", sorted(prefixes))

    # 2. Определяем входные параметры
    print("\n2. Входные параметры:")

    # Ищем первый сверточный слой
    first_conv_key = None
    for k in state_dict.keys():
        if 'proj.weight' in k or ('conv' in k and 'weight' in k and state_dict[k].dim() == 4):
            first_conv_key = k
            break

    if first_conv_key:
        shape = state_dict[first_conv_key].shape
        print(f"  Первый слой ({first_conv_key}): {shape}")
        print(f"  → Входные каналы (inp_channels) = {shape[1]}")
        print(f"  → Начальная размерность (dim) = {shape[0]}")
        inp_channels = shape[1]
        initial_dim = shape[0]
    else:
        print("  Не найден первый сверточный слой")
        inp_channels = None
        initial_dim = None

    # 3. Анализируем архитектуру по уровням
    print("\n3. Размерность каналов по уровням:")

    # Собираем информацию по уровням
    level_dims = {}
    level_blocks = {}

    for k, v in state_dict.items():
        # Ищем normalization слои для определения размерности
        if 'norm' in k and 'weight' in k and v.dim() == 1:
            # Определяем уровень
            if '_level1' in k:
                level = 'encoder_level1'
                level_dims[level] = v.shape[0]
                # Определяем номер блока
                parts = k.split('_level1.')[1].split('.')
                if parts[0].isdigit():
                    block_num = int(parts[0])
                    level_blocks[level] = max(level_blocks.get(level, 0), block_num + 1)

            elif '_level2' in k:
                level = 'encoder_level2'
                level_dims[level] = v.shape[0]
                parts = k.split('_level2.')[1].split('.')
                if parts[0].isdigit():
                    block_num = int(parts[0])
                    level_blocks[level] = max(level_blocks.get(level, 0), block_num + 1)

            elif '_level3' in k:
                level = 'encoder_level3'
                level_dims[level] = v.shape[0]
                parts = k.split('_level3.')[1].split('.')
                if parts[0].isdigit():
                    block_num = int(parts[0])
                    level_blocks[level] = max(level_blocks.get(level, 0), block_num + 1)

            elif k[0].isdigit() and '.norm' in k:  # latent блоки (0., 1., 2., 3.)
                level = 'latent'
                level_dims[level] = v.shape[0]
                block_num = int(k[0])
                level_blocks[level] = max(level_blocks.get(level, 0), block_num + 1)

            elif 'ent.' in k and '.norm' in k:  # decoder блоки
                level = 'decoder_level3'
                level_dims[level] = v.shape[0]
                parts = k.split('ent.')[1].split('.')
                if parts[0].isdigit():
                    block_num = int(parts[0])
                    level_blocks[level] = max(level_blocks.get(level, 0), block_num + 1)

    # Выводим информацию об уровнях
    level_order = ['encoder_level1', 'encoder_level2', 'encoder_level3', 'latent', 'decoder_level3']
    for level in level_order:
        if level in level_dims:
            blocks = level_blocks.get(level, '?')
            print(f"  {level:20} : {level_dims[level]:4} channels, {blocks} blocks")

    # 4. Определяем downsample/upsample
    print("\n4. Downsample/Upsample слои:")
    for k, v in state_dict.items():
        if v.dim() == 4:  # Conv слои
            if '.body.0.weight' in k:
                print(f"  Downsample: {k} -> {v.shape}")
            elif 'ody.0.weight' in k:
                print(f"  Upsample: {k} -> {v.shape}")
            elif 'chan_level' in k:
                print(f"  Channel reduction: {k} -> {v.shape}")

    # 5. Heads из temperature
    print("\n5. Attention heads:")
    for k, v in state_dict.items():
        if 'temperature' in k:
            print(f"  {k}: shape {tuple(v.shape)} -> {v.shape[0]} heads")

    # 6. Выходной слой
    print("\n6. Выходной слой:")
    for k, v in state_dict.items():
        if k == 'weight' and v.dim() == 4:  # Последний conv слой
            print(f"  output layer: {k} -> {v.shape}")
            print(f"  → Выходные каналы (out_channels) = {v.shape[0]}")
            out_channels = v.shape[0]
            break

    # 7. Предполагаемая конфигурация
    print("\n=== ПРЕДПОЛАГАЕМАЯ КОНФИГУРАЦИЯ МОДЕЛИ ===")

    if initial_dim and 'encoder_level1' in level_dims:
        # Определяем num_blocks
        num_blocks = [
            level_blocks.get('encoder_level1', 2),
            level_blocks.get('encoder_level2', 3),
            level_blocks.get('encoder_level3', 3),
            level_blocks.get('latent', 4)
        ]

        # Определяем heads (по умолчанию)
        heads = [1, 2, 4, 8]

        config = f"""
model = restormer_arch.Restormer(
    inp_channels={inp_channels},
    out_channels={out_channels if 'out_channels' in locals() else 3},
    dim={initial_dim},
    num_blocks={num_blocks},
    num_refinement_blocks={level_blocks.get('decoder_level3', 4)},
    heads={heads},
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type='WithBias',
    dual_pixel_task={inp_channels == 6 if inp_channels else False}
)
"""
        print(config)

    # 8. Для отладки: выводим все ключи сгруппированные
    print("\n=== ВСЕ КЛЮЧИ ПО ГРУППАМ ===")

    groups = {
        'patch_embed': [],
        'encoder_level1': [],
        'encoder_level2': [],
        'encoder_level3': [],
        'latent': [],
        'decoder': [],
        'downsample': [],
        'upsample': [],
        'output': [],
        'other': []
    }

    for k in state_dict.keys():
        if 'mbed' in k or 'proj' in k:
            groups['patch_embed'].append(k)
        elif '_level1' in k:
            groups['encoder_level1'].append(k)
        elif '_level2' in k:
            groups['encoder_level2'].append(k)
        elif '_level3' in k:
            groups['encoder_level3'].append(k)
        elif k[0].isdigit() and k[1] == '.':
            groups['latent'].append(k)
        elif 'ent.' in k:
            groups['decoder'].append(k)
        elif '.body.0.weight' in k:
            groups['downsample'].append(k)
        elif 'ody.0.weight' in k:
            groups['upsample'].append(k)
        elif k in ['weight', 'nv.weight']:
            groups['output'].append(k)
        else:
            groups['other'].append(k)

    for group_name, keys in groups.items():
        if keys:
            print(f"\n{group_name.upper()} ({len(keys)} параметров):")
            for k in sorted(keys)[:5]:  # первые 5
                if k in state_dict:
                    shape = tuple(state_dict[k].shape) if hasattr(state_dict[k], 'shape') else type(state_dict[k])
                    print(f"  {k} -> {shape}")
            if len(keys) > 5:
                print(f"  ... и еще {len(keys) - 5} параметров")

    return state_dict


# Запускаем анализ
state_dict = analyze_checkpoint("/home/linkoln-xd/python_projects/DocRes-master/checkpoints/experiment_name/33000.pkl")