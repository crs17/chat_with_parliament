from backend.preproces import PARTIPROGRAM_PATHS, process_partiprogram


for party_id in PARTIPROGRAM_PATHS:
    print(f'Processing party id: {party_id}')
    process_partiprogram(party_id)
    
