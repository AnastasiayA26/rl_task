# RL for gridworld

## Архитектура 
```
├── .gitignore
├── requirements.txt
├── README.md
├── environment/
| ├── __init__.py
│ ├── my_gridworld.py # реализация базовой среды
│ └── utils.py # визуализация базовой среды
├── model/
| ├── __init__.py
│ ├── a2c.py # актор-критик
│ └── dqn_lstm.py 
├── notebook/
│ ├── a2c_model.py # pipeline обучения актор-критик
│ └── dqn_lstm.py # pipeline обучения dqn
├── test/
  ├── __init__.py
  └── test_env.py # тестовое создание сред




## Детали реализации

###  Добавила штрафы за повторное посещение полей, так как агент зацикливался при обучении,  добавила lstm для dqn
### При тестовом создании среды (test_env.py) - результат записывается в папку env_examples
### Результаты обучения для алгоритмов записываются notebook/results для dqn и notebook/results_a2c для a2c_model

