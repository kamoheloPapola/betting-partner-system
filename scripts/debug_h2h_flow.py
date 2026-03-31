"""Debug H2H feature flow"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())

from src.core.container import ServiceContainer
from src.ml.distributions import ZeroInflatedEngine

# Get the match
pipeline = ServiceContainer.get_instance().pipeline
df = pipeline.run(league='SA')

juve = df[(df['home_team'].str.contains('JUVENTUS', na=False)) & (df['away_team'].str.contains('CREMONESE', na=False))]
match = juve.iloc[-1]

# Simulate what _calc_cards does
h2h_match_count = match.get('h2h_match_count', 0)
h2h_avg_cards = match.get('h2h_avg_cards')

print('=== DEBUGGING _calc_cards ===')
print(f'h2h_match_count: {h2h_match_count}')
print(f'h2h_avg_cards: {h2h_avg_cards}')
print(f'type(h2h_match_count): {type(h2h_match_count)}')

# Check condition
condition1 = h2h_match_count >= 2
condition2 = h2h_avg_cards is not None
condition3 = not pd.isna(h2h_avg_cards) if h2h_avg_cards is not None else False

print()
print('Condition checks:')
print(f'  h2h_match_count >= 2: {condition1}')
print(f'  h2h_avg_cards is not None: {condition2}')
print(f'  not pd.isna(h2h_avg_cards): {condition3}')

if condition1 and condition2 and condition3:
    print()
    print('H2H adjustment WOULD be applied!')
    h2h_weight = min(h2h_match_count / 5, 0.4)
    print(f'H2H weight: {h2h_weight}')
    
    # Simulate with mu=4.1
    mu_model = 4.1
    mu_blended = (1 - h2h_weight) * mu_model + h2h_weight * h2h_avg_cards
    print(f'mu_model: {mu_model} -> mu_blended: {mu_blended}')
    
    engine = ZeroInflatedEngine()
    res_before = engine.calculate_probabilities(mu_model, pi_zero=0.03)
    res_after = engine.calculate_probabilities(mu_blended, pi_zero=0.03)
    print(f'Cards O2.5 BEFORE: {res_before["cards_over_2_5"]*100:.1f}%')
    print(f'Cards O2.5 AFTER:  {res_after["cards_over_2_5"]*100:.1f}%')
else:
    print()
    print('H2H adjustment NOT applied - condition failed')
