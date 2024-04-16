# Generalized-simplicial-model
Generalized-simplicial-model is a configuration model for higher-order networks. The model provides tools for generating and dismantling 2-simplices within networks, along with associated functionalities for network analysis. The project includes implementation of a generalized simplicial model in the article "A Generalized Simplicial Model and Its Application", and also includes relevant datasets used in the research.
## Usage
To generate the simplicial model, run the following code:
```
python configuration_model.py
```
Parameters:

network_name: Name of your network or dataset.

dataset_path: Path to your dataset file. The dataset should be in the format of an undirected graph.

operation: Set operation to 0 for generating 2-simplices. Set operation to 1 for dismantling existing 2-simplices.

save_interval: The interval for saving the model during iterations.

experiment_repeats: The number of times the experiment should be repeated.

## Citation
If you use this algorithm in your research, please cite this project.
```
Rongmei Yang, Fang Zhou, Bo Liu, Linyuan Lü; A generalized simplicial model and its application. Chaos 1 April 2024; 34 (4): 043113.
```

## References
The four empirical networks come from the following references:
1. Watts, D. & Strogatz, S. Collective dynamics of small world networks. Nature 393, 440–2 (1998).
2. Guimer`a, R., Danon, L., Diaz-Guilera, A., Giralt, F. & Arenas, A. Self-similar community structure in a network of human interactions.
Physical review. E, Statistical, nonlinear, and soft matter physics 68, 065103 (2004).
3. Rossi, R. A. & Ahmed, N. K. The network data repository with interactive graph analytics and visualization. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, 4292–4293 (2015).
4. Schellenberger, J., Park, J., Conrad, T. & Palsson, B. Schellenberger j, park jo, conrad tm, palsson bo.. biochemical genetic and genomic
knowledgebase of large scale metabolic reconstructions. BMC bioinformatics 11, 213 (2010).
