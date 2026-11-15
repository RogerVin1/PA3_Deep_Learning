"""PA3 — Poucos rótulos: fauna em câmeras trap (ENA24) com dados não rotulados."""

# Referência do monitor: ResNet-18 (ImageNet) com todos os rótulos do train,
# época escolhida no val, F1 macro no test_public. É o "teto" da Parte 2.
F1_TETO_MONITOR = 0.686

# Limite do enunciado para o modelo entregue.
MAX_PARAMS = 12_000_000

# Orçamentos de rótulos por classe (rotulos_1/5/20.csv). O padrão é 5.
ORCAMENTOS = (1, 5, 20)
ORCAMENTO_PADRAO = 5
