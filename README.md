
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- Placeholder logo – replace with your own if desired -->
  <a href="https://github.com/your_username/QSPHAgent">
    <img src="images/logo.png" alt="QSPHAgent Logo" width="80" height="80">
  </a>

  <h3 align="center">QSPHAgent: Qualitative Structure–Property Hypothesis Agent</h3>

  <p align="center">
    A multi-agent framework for qualitative density-of-states (DOS) reasoning with RAG.
    <br />
    <a href="https://github.com/your_username/QSPHAgent"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_username/QSPHAgent">View Demo</a>
    ·
    <a href="https://github.com/your_username/QSPHAgent/issues">Report Bug</a>
    ·
    <a href="https://github.com/your_username/QSPHAgent/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- Placeholder figure – you can replace the target link and image reference -->
[![QSPHAgent Workflow Placeholder][product-screenshot]](docs/qsphagent_workflow.png)

QSPHAgent is a compact, research-oriented framework for generating **qualitative density-of-states (DOS) hypotheses** from crystal structures using a combination of:

- Materials Project data (CIF + DOS)  
- Physically motivated structural and DOS descriptors  
- Retrieval-augmented generation (RAG) over similar structures  
- A generator–critic **multi-agent loop** (LangGraph)  
- Simple KNN-based quantitative DOS prediction for grounding  
- Accuracy and semantic-similarity evaluation tools

The core idea: given **only the structure** and a **context set of similar materials**, QSPHAgent infers:

- Material classification (metallic / semiconducting / insulating / pseudogapped)  
- Qualitative DOS shape (e.g., U-shaped, flat, asymmetric)  
- Asymmetry around the Fermi level  
- Positions and relative intensities of dominant valence and conduction peaks  
- A structured reasoning trace for each prediction  

The system is notebook-friendly and designed as a starting point for **agentic XAI workflows** in materials science.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This project is implemented in Python and relies on a small set of core libraries:

* Python 3.10+
* [LangGraph](https://github.com/langchain-ai/langgraph)
* [LangChain OpenAI](https://github.com/langchain-ai/langchain)
* [Pydantic](https://docs.pydantic.dev/)
* [scikit-learn](https://scikit-learn.org/)
* [Matplotlib](https://matplotlib.org/)
* [pymatgen](https://pymatgen.org/)
* [mp-api](https://github.com/materialsproject/api)
* [python-dotenv](https://github.com/theskumar/python-dotenv)
* [PyYAML](https://pyyaml.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This section explains how to set up QSPHAgent locally and run the basic demonstration workflow.

### Prerequisites

You will need:

* Python 3.10+  
* A [Materials Project](https://materialsproject.org/open) API key (`MP_API_KEY`)  
* An [OpenAI](https://platform.openai.com/) API key (`OPENAI_API_KEY`)

Install Python packages (example):

```sh
pip install \
  langgraph \
  langchain-openai \
  pydantic \
  scikit-learn \
  matplotlib \
  python-dotenv \
  pymatgen \
  mp-api \
  pyyaml
````

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/your_username/QSPHAgent.git
   cd QSPHAgent
   ```

2. Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY="YOUR_OPENAI_KEY"
   MP_API_KEY="YOUR_MATERIALS_PROJECT_KEY"
   ```

3. Create or edit `config.yaml` to control database generation and DOS settings:

   ```yaml
   dos_features:
     smoothing: 1.0

   database:
     cutoff: 5.0
     output_file: "materials.json"
     primitive: true
     species: ["Si", "B", "N"]   # example; can be ["Si"], ["B","N"], ["Ga","N"], etc.
   ```

4. (Optional) Install the project as a local package:

   ```sh
   pip install -e .
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

Below is a minimal end-to-end example that:

1. Loads keys + config
2. Builds (or reuses) a local structural + DOS database from Materials Project
3. Splits into train/test sets
4. Trains a simple DOS predictor
5. Runs the generator–critic agent loop on one test material
6. Prints a structured comparison between ground truth DOS description and hypothesized DOS

```python
from qsph_agent.utils import load_keys_and_config
from qsph_agent.rag_data import build_database_if_needed, load_and_split_database
from qsph_agent.agent_workflow import (
    build_predictor,
    build_qsph_graph,
    make_initial_state,
    summarize_final_state,
)

# 1) Load keys + YAML config
MP_API_KEY, CONFIG = load_keys_and_config("config.yaml")

# 2) Build / load database based on species in config.yaml
datafile = build_database_if_needed(MP_API_KEY, CONFIG)

# 3) Split into train / test
train_set, test_set, full_data = load_and_split_database(datafile)

# 4) Build shallow quantitative DOS predictor (KNN on structural features)
predictor = build_predictor(train_set, n_neighbors=6)

# 5) Choose a test material
test_key = list(test_set.keys())[0]
test_entry = test_set[test_key]

# 6) Build the LangGraph workflow and initial state
graph = build_qsph_graph()
state = make_initial_state(test_entry, train_set, predictor, top_k=7)

# 7) Run the generator–critic loop (e.g., 2 iterations)
final_state = graph.invoke(state)

# 8) Print concise summary:
#    - structural summary
#    - ground truth DOS description
#    - hypothesized DOS description
summarize_final_state(final_state)
```

You can also loop over the entire test set and compute:

* classification accuracy on `material_classification`
* semantic similarity on `overall_dos_shape` and `asymmetry_comment`
  using OpenAI embeddings + cosine similarity.

*For more in-depth examples (e.g., Si vs BN vs GaN benchmarks), you can create a `notebooks/demo.ipynb` and copy this workflow.*

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

* [x] Basic Materials Project → structured DOS database
* [x] Structural + DOS feature extraction and description
* [x] RAG over structurally similar materials
* [x] Generator–Critic agent loop with LangGraph
* [x] KNN-based quantitative DOS grounding
* [x] Simple evaluation: accuracy + semantic similarity
* [ ] Add support for more complex DOS descriptors (orbital projections, spin)
* [ ] Extend to additional property channels (e.g., transport, magnetism)
* [ ] Add CLI entry point
* [ ] Add richer visualization of agent reasoning traces
* [ ] Add unit tests and CI workflow

See the [open issues](https://github.com/your_username/QSPHAgent/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are welcome.

If you have ideas to improve the agents, descriptors, or evaluation:

1. Fork the Project
2. Create your Feature Branch

   ```sh
   git checkout -b feature/AmazingFeature
   ```
3. Commit your Changes

   ```sh
   git commit -m "Add some AmazingFeature"
   ```
4. Push to the Branch

   ```sh
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

Please also feel free to open issues tagged `"enhancement"` or `"bug"` to discuss changes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Your Name (e.g., **Suvo Banik**) – [your_email@example.com](mailto:your_email@example.com)

Project Link: [https://github.com/your_username/QSPHAgent](https://github.com/your_username/QSPHAgent)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

Some key tools and resources that enabled this project:

* [Materials Project](https://materialsproject.org/)
* [mp-api](https://github.com/materialsproject/api)
* [pymatgen](https://pymatgen.org/)
* [LangGraph](https://github.com/langchain-ai/langgraph)
* [LangChain](https://github.com/langchain-ai/langchain)
* [OpenAI](https://platform.openai.com/)
* [scikit-learn](https://scikit-learn.org/)
* [Matplotlib](https://matplotlib.org/)
* [python-dotenv](https://github.com/theskumar/python-dotenv)
* [PyYAML](https://pyyaml.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/your_username/QSPHAgent.svg?style=for-the-badge
[contributors-url]: https://github.com/your_username/QSPHAgent/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/your_username/QSPHAgent.svg?style=for-the-badge
[forks-url]: https://github.com/your_username/QSPHAgent/network/members
[stars-shield]: https://img.shields.io/github/stars/your_username/QSPHAgent.svg?style=for-the-badge
[stars-url]: https://github.com/your_username/QSPHAgent/stargazers
[issues-shield]: https://img.shields.io/github/issues/your_username/QSPHAgent.svg?style=for-the-badge
[issues-url]: https://github.com/your_username/QSPHAgent/issues
[license-shield]: https://img.shields.io/github/license/your_username/QSPHAgent.svg?style=for-the-badge
[license-url]: https://github.com/your_username/QSPHAgent/blob/master/LICENSE.txt


```
::contentReference[oaicite:0]{index=0}
```
