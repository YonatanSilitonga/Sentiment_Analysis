# Sentiment Service Quickstart

1. Create and activate the virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Start the Flask API.

```bash
python app.py
```

4. Test the setup.

```bash
python test_setup.py
```

5. Call the API from Laravel.

```php
Http::post('http://127.0.0.1:5000/api/v1/predict', [
    'review_id' => 1,
    'text' => 'Tempatnya bagus tapi toiletnya kotor.',
]);
```

6. Use the response fields in Laravel.

- `success`
- `data.label`
- `data.confidence`
- `data.scores`
- `data.reason`
- `data.processed_text`
