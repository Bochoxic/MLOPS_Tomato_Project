Tomato disease project
==============================

Vicente Bosch & Paloma Guiral & Víctor Vigara


Project Organization
------------
`

│    .dvcignore

│   .gitignore

│   api.dockerfile

│   cloudbuild.yaml

│   data.dvc

│   Dockerfile

│   image.jpg

│   LICENSE

│   Makefile

│   prueba.JPG

│   README.md

│   requirements.txt

│   requirements_api.txt

│   setup.py

│   test.dockerfile

│   test_environment.py

│   tox.ini

│

├───.dvc

│       .gitignore

│       config

│

├───.github

│   └───workflows

│           flake8.yaml

│           isort.yml

│           tests.yml

│

├───config

│   │   default_config.yaml

│   │   wandb_sweep.yaml

│   │
│   └───experiment

│           exp1.yaml

│           exp2.yaml

│

├───docs

│       commands.rst

│       conf.py

│       getting-started.rst

│       index.rst

│       make.bat

│       Makefile

│

├───models

│       .gitkeep

│

├───notebooks

│       .gitkeep

│

├───references

│       .gitkeep

│

├───reports

│   │   .gitkeep

│   │   README.md

│   │   report.html

│   │   report.py

│   │
│   └───figures

│           .gitkeep

│           MLOperationsPipeline.jpeg

│           question19.jpeg

│           question20.jpeg

│           question21.jpeg

│

├───src

│   │   __init__.py

│   │
│   ├───api

│   │   └───app

│   │           main.py

│   │

│   ├───data

│   │       .gitkeep

│   │       download_data.py

│   │       make_dataset.py


│   │       __init__.py

│   │

│   ├───features

│   │       .gitkeep

│   │       build_features.py

│   │       __init__.py

│   │

│   ├───models

│   │       .gitkeep

│   │       model.py

│   │       model_ffnn.py

│   │       predict_model.py

│   │       train_functions.py

│   │       train_model_normal.py

│   │       train_model_sweep.py

│   │       __init__.py

│   │

│   └───visualization

│           .gitkeep

│           visualize.py

│           __init__.py

│

└───tests

        test_data.py
        
        test_model.py
        
        test_training.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
