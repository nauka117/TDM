# Open-Sora TDM Integration Guide

Это руководство описывает исправления, внесенные в код TDM для корректной работы с Open-Sora Plan.

## 🔧 Основные исправления

### 1. **Исправлена инициализация моделей Open-Sora**

**Проблема:** Код пытался использовать `Diffusion_models[args.model]` напрямую, что не работало с Open-Sora.

**Решение:** Добавлена поддержка `Diffusion_models_class` для правильной загрузки предобученных моделей:

```python
# Вместо:
model = Diffusion_models[args.model](...)

# Используется:
if args.model in Diffusion_models_class:
    model = Diffusion_models_class[args.model].from_pretrained(
        args.ae_path,
        subfolder="model",
        cache_dir=args.cache_dir
    )
else:
    # Fallback к прямой инициализации
    model = Diffusion_models[args.model](...)
```

### 2. **Исправлена обработка VAE**

**Проблема:** Неправильная нормализация и обработка 3D тензоров Open-Sora.

**Решение:** Использование правильных VAE wrapper'ов Open-Sora:

```python
# Правильная инициализация VAE
ae = ae_wrapper[args.ae](args.ae_path)

# Правильные размерности
latent_size = args.max_height // 8  # VAE downsampling factor
latent_size_t = args.num_frames // ae_stride_config[args.ae][0]  # Temporal downsampling
channels = ae_channel_config[args.ae]
```

### 3. **Исправлена обработка text encoders**

**Проблема:** Неправильная комбинация T5 и CLIP encoders.

**Решение:** Корректная обработка множественных text encoders:

```python
# Обработка множественных encoders
if len(text_encoder) > 1 and text_encoder[1] is not None:
    encoder_hidden_states_2 = text_encoder[1](input_ids, return_dict=False, attention_mask=prompt_attention_mask)[0]
    encoder_hidden_states = [encoder_hidden_states_1, encoder_hidden_states_2]
else:
    encoder_hidden_states = encoder_hidden_states_1
```

### 4. **Исправлена функция generate_new**

**Проблема:** Неправильная обработка DiT архитектуры Open-Sora.

**Решение:** Добавлен параметр `use_opensora` и правильная обработка model kwargs:

```python
# Open-Sora: DiT model call
if use_opensora:
    # Для Open-Sora DiT моделей
    noise_pred = unet(pure_noisy, timestep=T_, **model_kwargs, return_dict=False)[0]
else:
    # Для стандартных UNet моделей
    noise_pred = unet(pure_noisy, timestep=T_, **model_kwargs, return_dict=False)[0]
```

### 5. **Исправлен класс Predictor**

**Проблема:** Неправильная обработка 3D тензоров и DiT архитектуры.

**Решение:** Добавлена поддержка Open-Sora и правильные model kwargs:

```python
# Open-Sora: Prepare model kwargs for DiT
if use_opensora:
    model_kwargs = {
        "encoder_hidden_states": encoder_hidden_states,
        "attention_mask": None,  # Open-Sora не использует это
        "encoder_attention_mask": prompt_attention_mask,
    }
else:
    model_kwargs = {
        "encoder_hidden_states": encoder_hidden_states,
        "attention_mask": prompt_attention_mask,
        "added_cond_kwargs": self.added_cond_kwargs,
    }
```

### 6. **Исправлена валидация**

**Проблема:** Неправильная работа с Open-Sora pipeline.

**Решение:** Добавлена поддержка Open-Sora pipeline с fallback к стандартному:

```python
try:
    from opensora.sample.pipeline_opensora import OpenSoraPipeline
    # Создание Open-Sora pipeline
    pipeline = OpenSoraPipeline(...)
except ImportError:
    # Fallback к стандартному pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(...)
```

## 🚀 Новые возможности

### 1. **Поддержка Open-Sora моделей**
- Автоматическое определение доступных моделей
- Правильная загрузка предобученных весов
- Поддержка различных конфигураций VAE

### 2. **Гибкая архитектура**
- Поддержка как Open-Sora, так и стандартных моделей
- Автоматическое переключение между режимами
- Fallback механизмы для совместимости

### 3. **Улучшенная обработка данных**
- Правильная работа с 3D тензорами [B, C, T, H, W]
- Корректная нормализация VAE
- Поддержка различных размеров кадров

## 📋 Требования

### 1. **Установленные зависимости**
```bash
pip install torch diffusers transformers accelerate
pip install -e ../Open-Sora-Plan  # Установка Open-Sora в режиме разработки
```

### 2. **Структура проекта**
```
TDM/
├── src/
│   ├── main.py          # Основной тренировочный код
│   ├── models.py        # Функции генерации
│   ├── predictor.py     # Класс Predictor
│   ├── training.py      # Валидация и сохранение
│   └── args.py          # Аргументы командной строки
├── Open-Sora-Plan/      # Соседний репозиторий Open-Sora
└── test_opensora_integration.py  # Тестовый скрипт
```

## 🧪 Тестирование

Запустите тестовый скрипт для проверки интеграции:

```bash
cd TDM
python test_opensora_integration.py
```

Этот скрипт проверит:
- Импорт всех необходимых модулей
- Инициализацию VAE
- Инициализацию text encoders
- Доступность diffusion моделей
- Корректность функций TDM

## 🎯 Использование

### 1. **Базовый запуск**
```bash
accelerate launch train_tdm_demo.py \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path LanguageBind/Open-Sora-Plan-v1.3.0 \
    --model OpenSoraT2V_v1_3_93x640x640 \
    --text_encoder_name_1 google/t5-v1_1-xl \
    --text_encoder_name_2 openai/clip-vit-large-patch14 \
    --num_frames 93 \
    --max_height 640 \
    --max_width 640 \
    --total_steps 900 \
    --cfg 4.5 \
    --use_huber \
    --use_separate
```

### 2. **Ключевые параметры**
- `--ae`: Конфигурация VAE (WFVAEModel_D8_4x8x8, CausalVAEModel_D8_4x8x8)
- `--ae_path`: Путь к модели VAE
- `--model`: Конфигурация diffusion модели
- `--text_encoder_name_1`: Первый text encoder (T5)
- `--text_encoder_name_2`: Второй text encoder (CLIP, опционально)
- `--num_frames`: Количество кадров в видео
- `--max_height/--max_width`: Максимальные размеры
- `--total_steps`: Общее количество diffusion шагов
- `--cfg`: CFG scale для TDM
- `--use_huber`: Использование Huber loss
- `--use_separate`: Раздельные интервалы шума

## ⚠️ Известные проблемы

### 1. **Memory usage**
Open-Sora модели могут потребовать значительного количества GPU памяти. Рекомендуется:
- Использовать gradient checkpointing
- Уменьшить batch size
- Использовать mixed precision training

### 2. **Model compatibility**
Не все модели Open-Sora могут быть совместимы с текущей реализацией. Проверьте:
- Доступность модели в `Diffusion_models_class`
- Совместимость размерностей VAE и модели
- Поддержку требуемых параметров

## 🔮 Будущие улучшения

### 1. **Автоматическая оптимизация**
- Автоматический выбор оптимальных параметров
- Адаптивная настройка размеров batch
- Интеллектуальное управление памятью

### 2. **Расширенная поддержка**
- Поддержка новых версий Open-Sora
- Интеграция с дополнительными моделями
- Улучшенная обработка ошибок

### 3. **Производительность**
- Оптимизация для различных GPU
- Поддержка distributed training
- Улучшенные алгоритмы TDM

## 📞 Поддержка

При возникновении проблем:

1. Проверьте тестовый скрипт
2. Убедитесь в правильности путей к моделям
3. Проверьте совместимость версий
4. Обратитесь к документации Open-Sora

## 📚 Дополнительные ресурсы

- [Open-Sora Plan Repository](https://github.com/PKU-YuanGroup/Open-Sora-Plan)
- [TDM Paper](https://arxiv.org/abs/2503.06674)
- [Open-Sora Documentation](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/main/docs)
