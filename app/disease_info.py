"""
Disease Information and Treatment Recommendations
"""

DISEASE_INFO = {
    'orf': {
        'name': 'Orf (Contagious Ecthyma)',
        'description': 'Viral infection causing scabby sores around mouth, lips, and nose.',
        'symptoms': [
            'Scabs and lesions around mouth and lips',
            'Difficulty eating and drinking',
            'Weight loss',
            'Secondary bacterial infections possible'
        ],
        'treatment': [
            'Isolate affected sheep immediately',
            'Apply antiseptic ointment to lesions',
            'Ensure clean water and soft feed available',
            'Consult veterinarian for severe cases',
            'Vaccination available for prevention'
        ],
        'severity': 'Moderate',
        'contagious': 'Yes - highly contagious to other sheep and humans'
    },
    
    'flystrike': {
        'name': 'Flystrike (Myiasis)',
        'description': 'Fly larvae (maggots) feeding on sheep tissue, potentially fatal if untreated.',
        'symptoms': [
            'Restlessness and tail switching',
            'Wool loss and discoloration',
            'Foul odor',
            'Visible maggots in wool',
            'Open wounds'
        ],
        'treatment': [
            'URGENT: Remove maggots immediately',
            'Clip wool around affected area',
            'Clean wounds with antiseptic',
            'Apply fly strike treatment (Cyromazine)',
            'Veterinary attention for severe cases',
            'Pain relief may be needed'
        ],
        'severity': 'High - Can be fatal',
        'contagious': 'No - but attracts more flies'
    },
    
    'ringworm': {
        'name': 'Ringworm (Dermatophytosis)',
        'description': 'Fungal infection causing circular patches of hair loss.',
        'symptoms': [
            'Circular patches of hair loss',
            'Scaly, crusty skin',
            'Itching and scratching',
            'Grey-white crusts on skin'
        ],
        'treatment': [
            'Isolate affected animals',
            'Apply antifungal topical treatment',
            'Improve ventilation in housing',
            'Disinfect equipment and housing',
            'Most cases self-resolve in 1-3 months',
            'Consult vet for persistent cases'
        ],
        'severity': 'Low to Moderate',
        'contagious': 'Yes - to other animals and humans'
    },
    
    'sheep_scab': {
        'name': 'Sheep Scab (Psoroptic Mange)',
        'description': 'Mite infestation causing intense itching and wool loss.',
        'symptoms': [
            'Intense itching and scratching',
            'Wool loss and damage',
            'Thickened, crusty skin',
            'Restlessness',
            'Weight loss',
            'Secondary infections possible'
        ],
        'treatment': [
            'Notify authorities (notifiable disease in many countries)',
            'Treat ALL sheep in flock with approved acaricide',
            'Quarantine new animals for 21 days',
            'Two treatments 10-14 days apart usually required',
            'MUST be treated by prescription products',
            'Veterinary supervision essential'
        ],
        'severity': 'High - Legally notifiable',
        'contagious': 'Yes - highly contagious'
    },
    
    'healthy': {
        'name': 'Healthy Sheep',
        'description': 'No visible signs of disease detected.',
        'symptoms': [
            'Clear eyes',
            'Good body condition',
            'Active and alert',
            'Healthy fleece',
            'Normal eating and drinking'
        ],
        'treatment': [
            'Continue regular health checks',
            'Maintain vaccination schedule',
            'Provide balanced nutrition',
            'Ensure clean water access',
            'Monitor for any changes in behavior',
            'Regular deworming as per schedule'
        ],
        'severity': 'None',
        'contagious': 'N/A'
    }
}

def get_disease_info(disease_name):
    """Get detailed information about a disease"""
    disease_key = disease_name.lower().replace(' ', '_')
    return DISEASE_INFO.get(disease_key, {
        'name': disease_name,
        'description': 'Information not available',
        'symptoms': [],
        'treatment': [],
        'severity': 'Unknown',
        'contagious': 'Unknown'
    })