from backend.preproces import process_partiprogram
from backend.common import PARTY_MANIFESTS

for party_id in PARTY_MANIFESTS:
    print(f'Processing party id: {party_id}')
    process_partiprogram(party_id)
    
