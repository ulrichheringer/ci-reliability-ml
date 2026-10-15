# ci-reliability-ml

Classifica causas de falha de CI por texto e estima risco de flaky tests a partir de sinais históricos tabulares.

## Dados

O pipeline gera dados sintéticos determinísticos localmente, identificados como sintéticos em `pipeline.py`. Nenhum dado privado ou download externo é necessário. A divisão e a semente de avaliação são descritas junto às métricas abaixo.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Métricas obtidas

Seed 42: holdout por templates para causas de CI — acurácia **0,8333**, F1 macro **0,8667** (72 logs); holdout estratificado para risco flaky — acurácia **0,9120**, F1 **0,9461**, ROC AUC **0,9874** (250 testes). São medições do conjunto sintético e não afirmam desempenho em dados de produção.

Limitação: dados e regras de geração simplificados; antes de uso real, substitua-os por dados com licença e proveniência adequadas e valide fora da amostra.
