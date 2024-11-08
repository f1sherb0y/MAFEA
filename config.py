# config.py

# Graph configurations
GRAPH_CONFIGS = {
    'Chain': {
        'nodes': [
            {'id': 1, 'capability': 1},
            {'id': 2, 'capability': 2},
            {'id': 3, 'capability': 3},
            {'id': 4, 'capability': 4},
            {'id': 5, 'capability': 5},
        ],
        'edges': [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ]
    },
    'Ring': {
        'nodes': [
            {'id': 1, 'capability': 1},
            {'id': 2, 'capability': 2},
            {'id': 3, 'capability': 3},
            {'id': 4, 'capability': 4},
            {'id': 5, 'capability': 5},
        ],
        'edges': [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 1),
        ]
    },
    'Star': {
        'nodes': [
            {'id': 1, 'capability': 5},  # Central agent with highest capability
            {'id': 2, 'capability': 1},
            {'id': 3, 'capability': 2},
            {'id': 4, 'capability': 3},
            {'id': 5, 'capability': 4},
        ],
        'edges': [
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
        ]
    },
    'FullyConnected': {
        'nodes': [
            {'id': 1, 'capability': 1},
            {'id': 2, 'capability': 2},
            {'id': 3, 'capability': 3},
            {'id': 4, 'capability': 4},
            {'id': 5, 'capability': 5},
        ],
        'edges': [
            (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 3), (2, 4), (2, 5),
            (3, 4), (3, 5),
            (4, 5),
        ]
    },
    'Random': {
        'nodes': [
            {'id': 1, 'capability': 1},
            {'id': 2, 'capability': 3},
            {'id': 3, 'capability': 2},
            {'id': 4, 'capability': 5},
            {'id': 5, 'capability': 4},
        ],
        'edges': [
            (1, 3),
            (2, 4),
            (3, 5),
            (1, 5),
            (1, 2),
        ]
    },
}

MAX_DEBATE_ROUNDS_PER_PAIR = 3 
