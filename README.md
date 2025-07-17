# Graph-based Entity Resolution and Schema Matching

This project implements a graph-based approach for entity resolution (ER) and schema matching (SM), likely utilizing a TGAE (Topological Graph Autoencoder) model.

## Project Structure

```
.
├── README.md                   # This file
├── args.py                     # Configuration file for setting up model and data parameters
├── efficient_kan/              # Directory for the KAN layer implementation
│   ├── __init__.py
│   └── kan.py
├── embeddings_quality.py       # Script to evaluate the quality of embeddings
├── entity_resolution.py        # Core logic for the entity resolution task
├── input_data.py               # Data loading and preprocessing utilities
├── kan_layer.py                # KAN layer implementation (likely used by model.py)
├── model.py                    # Definition of the graph neural network model (TGAE)
├── networkx_adj_features.py    # Helper script for networkx graph processing
├── schema_matching.py          # Core logic for the schema matching task
├── testEQ.py                   # Script to run embedding quality tests
├── testER.py                   # Script to run entity resolution tests
├── testSM.py                   # Script to run schema matching tests
├── train.py                    # Main script to train the model and generate embeddings
└── utils.py                    # Utility functions used across the project
```

## Dependencies

This project requires the following Python libraries:

- numpy
- scipy
- networkx
- pickle (pkl)
- Potentially a deep learning framework like TensorFlow or PyTorch (based on the model structure)

You can install them using pip:
```bash
pip install numpy scipy networkx
```

## How to Run

1.  **Configure Parameters:**
    - Open `args.py` to set the paths for your dataset, embeddings, and other model parameters.
    - Specify the `test_type` as either `'ER'` for Entity Resolution or `'SM'` for Schema Matching.

2.  **Train the Model:**
    - Run the `train.py` script to train the TGAE model on your data. This will generate an embeddings file.
    ```bash
    python train.py
    ```

3.  **Run Tests:**
    - **For Entity Resolution:** After training, run `testER.py` to evaluate the model on the ER task.
      ```bash
      python testER.py
      ```
    - **For Schema Matching:** Similarly, run `testSM.py` for the SM task.
      ```bash
      python testSM.py
      ```

Make sure all paths in `args.py` are correctly configured before running the scripts.