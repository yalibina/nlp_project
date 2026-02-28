# Информация о моделях бесплатных с https://openrouter.ai/models?max_price=0 и степени их надежности

## Работают довольно наджено
1. `openai/gpt-oss-120b:free`
2. `meta-llama/llama-3.3-70b-instruct:free`
3. `google/gemma-3-12b-it:free`
4. `google/gemma-3-27b-it:free`
5. `nvidia/nemotron-3-nano-30b-a3b:free`
6. `nvidia/nemotron-nano-9b-v2:free`
7. `z-ai/glm-4.5-air:free `

## Работают заметно менее стабильно (чаще падают с ошибкой )
1. `meta-llama/llama-3.2-3b-instruct:free`
2. `openai/gpt-oss-20b:free`
3. `qwen/qwen3-next-80b-a3b-instruct:free`
4. `qwen/qwen3-coder:free`
5. `qwen/qwen3-4b:free`

## Примечания

### Для всех мделей кроме gemma работает вот такой пайплайн

```python
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)
completion = client.chat.completions.create(
        model=model_name,
        messages=[
          {"role": "system", "content": "Ты полезный ИИ-ассистент. Отвечай на русском языке."},
          {"role": "user", "content": "Сделай сальто"}
        ],
)
print(completion.choices[0].message.content)
```

### Для gemma же чуть иначе

```python
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)
completion = client.chat.completions.create(
        model=model_name,
        messages=[
          {"role": "user", 
          "content": "Ты полезный ИИ-ассистент. Отвечай на русском языке. Сделай сальто"}
        ],
)
print(completion.choices[0].message.content)
```

api_key брать на сайте и не забыть разрешить делать все что угодно с вашей историей общения